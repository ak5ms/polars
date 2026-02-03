use arrow::array::builder::ShareStrategy;
use arrow::array::BinaryArray;
use arrow::array::Array;
use polars_core::chunked_array::ops::row_encode::_get_rows_encoded_arr;
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_core::series::builder::SeriesBuilder;
use polars_error::{PolarsResult, polars_ensure, polars_err};
use polars_utils::pl_str::PlSmallStr;

fn build_output_schema(dfs: &[DataFrame], keys: &[PlSmallStr]) -> PolarsResult<Schema> {
    let mut schema = Schema::with_capacity(dfs.iter().map(|df| df.width()).sum());
    let mut payload_names = PlHashSet::new();

    for key in keys {
        let dtype = dfs[0].column(key)?.dtype().clone();
        schema.with_column(key.clone(), dtype);
    }

    for (idx, df) in dfs.iter().enumerate() {
        for (name, dtype) in df.schema().iter() {
            if keys.contains(name) {
                continue;
            }
            polars_ensure!(
                payload_names.insert(name.clone()),
                Duplicate: "join_many received duplicate column name '{name}' in input {idx}"
            );
            schema.with_column(name.clone(), dtype.clone());
        }
    }

    Ok(schema)
}

fn encode_keys(df: &DataFrame, keys: &[PlSmallStr]) -> PolarsResult<BinaryArray<i64>> {
    let columns = keys
        .iter()
        .map(|name| df.column(name).cloned())
        .collect::<PolarsResult<Vec<_>>>()?;
    let descending = vec![false; keys.len()];
    let nulls_last = vec![false; keys.len()];
    _get_rows_encoded_arr(&columns, &descending, &nulls_last, true)
}

fn compare_keys(a: &BinaryArray<i64>, a_idx: usize, b: &BinaryArray<i64>, b_idx: usize) -> std::cmp::Ordering {
    let a_null = a.is_null(a_idx);
    let b_null = b.is_null(b_idx);
    match (a_null, b_null) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a.value(a_idx).cmp(b.value(b_idx)),
    }
}

fn keys_equal(a: &BinaryArray<i64>, a_idx: usize, b: &BinaryArray<i64>, b_idx: usize) -> bool {
    if a.is_null(a_idx) || b.is_null(b_idx) {
        return false;
    }
    a.value(a_idx) == b.value(b_idx)
}

fn append_key_column(
    builder: &mut SeriesBuilder,
    series: &Series,
    idx: usize,
    repeats: usize,
) {
    let slice = series.slice(idx as i64, 1);
    builder.subslice_extend_repeated(&slice, 0, 1, repeats, ShareStrategy::Always);
}

fn append_group_column(
    builder: &mut SeriesBuilder,
    series: &Series,
    start: usize,
    len: usize,
    pre: usize,
    post: usize,
) {
    for _ in 0..pre {
        if post == 1 {
            builder.subslice_extend(series, start, len, ShareStrategy::Always);
        } else {
            builder.subslice_extend_each_repeated(series, start, len, post, ShareStrategy::Always);
        }
    }
}

pub fn join_many_sorted(dfs: &[DataFrame], keys: &[PlSmallStr]) -> PolarsResult<DataFrame> {
    polars_ensure!(
        dfs.len() >= 2,
        NoData: "join_many expects at least two dataframes"
    );
    polars_ensure!(
        !keys.is_empty(),
        InvalidOperation: "join_many expects at least one key column"
    );

    let mut seen_keys = PlHashSet::with_capacity(keys.len());
    for key in keys {
        polars_ensure!(
            seen_keys.insert(key.clone()),
            Duplicate: "join_many received duplicate key '{key}'"
        );
    }

    let first_schema = dfs[0].schema();
    for (idx, df) in dfs.iter().enumerate() {
        for key in keys {
            polars_ensure!(
                df.schema().contains(key),
                ColumnNotFound: "join_many key '{key}' is missing from input {idx}"
            );
            polars_ensure!(
                df.column(key)?.dtype() == first_schema.get(key).unwrap(),
                SchemaMismatch: "join_many key '{key}' has mismatched types across inputs"
            );
        }
    }

    let output_schema = build_output_schema(dfs, keys)?;
    let mut builders = output_schema
        .iter_values()
        .map(|dtype| SeriesBuilder::new(dtype.clone()))
        .collect::<Vec<_>>();

    let encoded_keys = dfs
        .iter()
        .map(|df| encode_keys(df, keys))
        .collect::<PolarsResult<Vec<_>>>()?;

    let mut cursors = vec![0usize; dfs.len()];
    let heights = dfs.iter().map(|df| df.height()).collect::<Vec<_>>();

    let key_columns = dfs
        .iter()
        .map(|df| {
            let columns = df
                .select(keys)?
                .columns()
                .iter()
                .map(|col| col.as_materialized_series().clone())
                .collect::<Vec<_>>();
            Ok(columns)
        })
        .collect::<PolarsResult<Vec<Vec<Series>>>>()?;

    let mut output_columns = Vec::with_capacity(output_schema.len());
    for key in keys {
        output_columns.push((None, key.clone()));
    }
    for (df_idx, df) in dfs.iter().enumerate() {
        for name in df.get_column_names() {
            if keys.contains(name) {
                continue;
            }
            output_columns.push((Some(df_idx), name.clone()));
        }
    }

    while cursors
        .iter()
        .zip(&heights)
        .any(|(cursor, height)| *cursor < *height)
    {
        let mut min_df = None;
        for (df_idx, (cursor, height)) in cursors.iter().zip(&heights).enumerate() {
            if *cursor >= *height {
                continue;
            }
            min_df = match min_df {
                None => Some(df_idx),
                Some(current) => {
                    let ord = compare_keys(
                        &encoded_keys[df_idx],
                        *cursor,
                        &encoded_keys[current],
                        cursors[current],
                    );
                    if ord == std::cmp::Ordering::Less {
                        Some(df_idx)
                    } else {
                        Some(current)
                    }
                },
            };
        }

        let min_df = min_df.expect("at least one dataframe has remaining rows");
        let min_cursor = cursors[min_df];
        let min_is_null = encoded_keys[min_df].is_null(min_cursor);

        let mut matched = Vec::new();
        if min_is_null {
            matched.push(min_df);
        } else {
            for (df_idx, (cursor, height)) in cursors.iter().zip(&heights).enumerate() {
                if *cursor >= *height {
                    continue;
                }
                if keys_equal(
                    &encoded_keys[df_idx],
                    *cursor,
                    &encoded_keys[min_df],
                    min_cursor,
                ) {
                    matched.push(df_idx);
                }
            }
        }

        matched.sort_unstable();
        let mut group_ranges = vec![(0usize, 0usize); dfs.len()];
        for &df_idx in &matched {
            let start = cursors[df_idx];
            let mut end = start + 1;
            let height = heights[df_idx];
            while end < height
                && keys_equal(
                    &encoded_keys[df_idx],
                    end,
                    &encoded_keys[df_idx],
                    start,
                )
            {
                end += 1;
            }
            cursors[df_idx] = end;
            group_ranges[df_idx] = (start, end - start);
        }

        let total_rows = matched
            .iter()
            .map(|idx| group_ranges[*idx].1)
            .try_fold(1usize, |acc, len| acc.checked_mul(len))
            .ok_or_else(|| polars_err!(ComputeError: "join_many output size overflow"))?;

        let mut pre_products = vec![1usize; dfs.len()];
        let mut post_products = vec![1usize; dfs.len()];
        for (pos, df_idx) in matched.iter().enumerate() {
            let pre = matched[..pos]
                .iter()
                .map(|idx| group_ranges[*idx].1)
                .product::<usize>();
            let post = matched[pos + 1..]
                .iter()
                .map(|idx| group_ranges[*idx].1)
                .product::<usize>();
            pre_products[*df_idx] = pre;
            post_products[*df_idx] = post.max(1);
        }

        for (out_idx, (df_idx_opt, name)) in output_columns.iter().enumerate() {
            let builder = &mut builders[out_idx];
            match df_idx_opt {
                None => {
                    let key_idx = keys.iter().position(|k| k == name).unwrap();
                    let key_series = &key_columns[min_df][key_idx];
                    append_key_column(builder, key_series, min_cursor, total_rows);
                },
                Some(df_idx) => {
                    if matched.contains(df_idx) {
                        let (start, len) = group_ranges[*df_idx];
                        let series = dfs[*df_idx].column(name)?.as_materialized_series();
                        append_group_column(
                            builder,
                            series,
                            start,
                            len,
                            pre_products[*df_idx],
                            post_products[*df_idx],
                        );
                    } else {
                        builder.extend_nulls(total_rows);
                    }
                },
            }
        }
    }

    let columns = output_schema
        .iter_names()
        .zip(builders)
        .map(|(name, builder)| {
            let series = builder.freeze(name.clone());
            Column::from(series)
        })
        .collect::<Vec<_>>();

    Ok(unsafe { DataFrame::new_unchecked(columns[0].len(), columns) })
}
