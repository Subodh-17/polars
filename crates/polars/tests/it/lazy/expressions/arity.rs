use super::*;

#[test]
#[cfg(feature = "unique_counts")]
fn test_list_broadcast() {
    // simply test if this runs
    df![
        "g" => [1, 1, 1],
        "a" => [1, 2, 3],
    ]
    .unwrap()
    .lazy()
    .group_by([col("g")])
    .agg([col("a").unique_counts() * len()])
    .collect()
    .unwrap();
}

#[test]
fn ternary_expand_sizes() -> PolarsResult<()> {
    let df = df! {
        "a" => [Some("a1"), None, None],
        "b" => [Some("b1"), Some("b2"), None]
    }?;
    let out = df
        .lazy()
        .with_column(
            when(not(lit(true)))
                .then(lit("unexpected"))
                .when(not(col("a").is_null()))
                .then(col("a"))
                .when(not(col("b").is_null()))
                .then(col("b"))
                .otherwise(lit("otherwise"))
                .alias("c"),
        )
        .collect()?;
    let vals = out.column("c")?.str()?.no_null_iter().collect::<Vec<_>>();
    assert_eq!(vals, &["a1", "b2", "otherwise"]);
    Ok(())
}

#[test]
#[cfg(feature = "strings")]
fn includes_null_predicate_3038() -> PolarsResult<()> {
    let df = df! {
        "a" => [Some("a1"), None, None],
    }?;
    let res = df
        .lazy()
        .with_column(
            when(col("a").map(
                move |s| {
                    s.str()?
                        .to_lowercase()
                        .contains("not_exist", true)
                        .map(|ca| ca.into_column())
                },
                |_, f| Ok(Field::new(f.name().clone(), DataType::Boolean)),
            ))
            .then(lit("unexpected"))
            .when(col("a").eq(lit("a1".to_string())))
            .then(lit("good hit"))
            .otherwise(Expr::Literal(LiteralValue::untyped_null()))
            .alias("b"),
        )
        .collect()?;

    let exp_df = df! {
        "a" => [Some("a1"), None, None],
        "b" => [Some("good hit"), None, None],
    }?;
    assert!(res.equals_missing(&exp_df));

    let df = df! {
        "a" => ["a1", "a2", "a3", "a4", "a2"],
        "b" => [Some("tree"), None, None, None, None],
    }?;
    let res = df
        .lazy()
        .with_column(
            when(col("b").map(
                move |s| {
                    s.str()?
                        .to_lowercase()
                        .contains_literal("non-existent")
                        .map(|ca| ca.into_column())
                },
                |_, f| Ok(Field::new(f.name().clone(), DataType::Boolean)),
            ))
            .then(lit("weird-1"))
            .when(col("a").eq(lit("a1".to_string())))
            .then(lit("ok1"))
            .when(col("a").eq(lit("a2".to_string())))
            .then(lit("ok2"))
            .when(lit(true))
            .then(lit("ft"))
            .otherwise(Expr::Literal(LiteralValue::untyped_null()))
            .alias("c"),
        )
        .collect()?;
    let exp_df = df! {
        "a" => ["a1", "a2", "a3", "a4", "a2"],
        "b" => [Some("tree"), None, None, None, None],
        "c" => ["ok1", "ok2", "ft", "ft", "ok2"]
    }?;
    assert!(res.equals_missing(&exp_df));

    Ok(())
}

#[test]
#[cfg(feature = "dtype-categorical")]
fn test_when_then_otherwise_cats() -> PolarsResult<()> {
    let lf = df!["book" => [Some("bookA"),
        None,
        Some("bookB"),
        None,
        Some("bookA"),
        Some("bookC"),
        Some("bookC"),
        Some("bookC")],
        "user" => [Some("bob"), Some("bob"), Some("bob"), Some("tim"), Some("lucy"), Some("lucy"), None, None]
    ]?.lazy();

    let out = lf
        .with_column(col("book").cast(DataType::from_categories(Categories::global())))
        .with_column(col("user").cast(DataType::from_categories(Categories::global())))
        .with_column(
            when(col("book").is_null())
                .then(col("user"))
                .otherwise(col("book"))
                .alias("a"),
        )
        .collect()?;

    assert_eq!(
        out.column("a")?
            .cat32()?
            .iter_str()
            .flatten()
            .collect::<Vec<_>>(),
        &[
            "bookA", "bob", "bookB", "tim", "bookA", "bookC", "bookC", "bookC"
        ]
    );

    Ok(())
}

#[test]
fn test_when_then_otherwise_single_bool() -> PolarsResult<()> {
    let df = df![
        "key" => ["a", "b", "b"],
        "val" => [Some(1), Some(2), None]
    ]?;

    let out = df
        .lazy()
        .group_by_stable([col("key")])
        .agg([when(col("val").null_count().gt(lit(0)))
            .then(Null {}.lit())
            .otherwise(col("val").sum())
            .alias("sum_null_prop")])
        .collect()?;

    let expected = df![
        "key" => ["a", "b"],
        "sum_null_prop" => [Some(1), None]
    ]?;

    assert!(out.equals_missing(&expected));

    Ok(())
}

#[test]
#[cfg(feature = "unique_counts")]
fn test_update_groups_in_cast() -> PolarsResult<()> {
    let df = df![
        "group" =>  ["A" ,"A", "A", "B", "B", "B", "B"],
        "id"=> [1, 2, 1, 4, 5, 4, 6],
    ]?;

    // optimized to
    // col("id").unique_counts().cast(int64) * -1
    // in aggregation that cast coerces a list and the cast may forget to update groups
    let out = df
        .lazy()
        .group_by_stable([col("group")])
        .agg([col("id").unique_counts() * lit(-1)])
        .collect()?;

    let expected = df![
        "group" =>  ["A" ,"B"],
        "id"=> [AnyValue::List(Series::new("".into(), [-2i64, -1])), AnyValue::List(Series::new("".into(), [-2i64, -1, -1]))]
    ]?;

    assert!(out.equals(&expected));
    Ok(())
}

#[test]
fn test_when_then_otherwise_sum_in_agg() -> PolarsResult<()> {
    let df = df![
        "groups" => [1, 1, 2, 2],
        "dist_a" => [0.1, 0.2, 0.5, 0.5],
        "dist_b" => [0.8, 0.2, 0.5, 0.2],
    ]?;

    let q = df
        .lazy()
        .group_by([col("groups")])
        .agg([
            when(all().exclude_cols(["groups"]).as_expr().sum().eq(lit(1)))
                .then(all().exclude_cols(["groups"]).as_expr().sum())
                .otherwise(lit(LiteralValue::untyped_null())),
        ])
        .sort(["groups"], Default::default());

    let expected = df![
        "groups" => [1, 2],
        "dist_a" => [None, Some(1.0f64)],
        "dist_b" => [Some(1.0f64), None]
    ]?;
    assert!(q.collect()?.equals_missing(&expected));

    Ok(())
}

#[test]
fn test_binary_over_3930() -> PolarsResult<()> {
    let df = df![
        "class" => ["a", "a", "a", "b", "b", "b"],
        "score" => [0.2, 0.5, 0.1, 0.3, 0.4, 0.2]
    ]?;

    let ss = col("score").pow(2);
    let mdiff = (ss.clone().shift(lit(-1)) - ss.shift(lit(1))) / lit(2);
    let out = df
        .lazy()
        .select([mdiff.over([col("class")]).unwrap()])
        .collect()?;

    let out = out.column("score")?;
    let out = out.f64()?;

    assert_eq!(
        Vec::from(out),
        &[
            None,
            Some(-0.015000000000000003),
            None,
            None,
            Some(-0.024999999999999994),
            None
        ]
    );

    Ok(())
}

#[test]
#[cfg(feature = "rank")]
fn test_ternary_aggregation_set_literals() -> PolarsResult<()> {
    let df = df![
        "name" => ["a", "b", "a", "b"],
        "value" => [1, 3, 2, 4]
    ]?;

    let out = df
        .clone()
        .lazy()
        .group_by([col("name")])
        .agg([when(col("value").sum().eq(lit(3)))
            .then(col("value").rank(Default::default(), None))
            .otherwise(lit(Series::new("".into(), &[10 as IdxSize])))])
        .sort(["name"], Default::default())
        .collect()?;

    let out = out.column("value")?;
    assert_eq!(
        out.get(0)?,
        AnyValue::List(Series::new("".into(), &[1 as IdxSize, 2 as IdxSize]))
    );
    assert_eq!(
        out.get(1)?,
        AnyValue::List(Series::new("".into(), &[10 as IdxSize, 10 as IdxSize]))
    );

    let out = df
        .clone()
        .lazy()
        .group_by([col("name")])
        .agg([when(col("value").sum().eq(lit(3)))
            .then(lit(Series::new("".into(), &[10 as IdxSize])).alias("value"))
            .otherwise(col("value").rank(Default::default(), None))])
        .sort(["name"], Default::default())
        .collect()?;

    let out = out.column("value")?;
    assert_eq!(
        out.get(1)?,
        AnyValue::List(Series::new("".into(), &[1 as IdxSize, 2]))
    );
    assert_eq!(
        out.get(0)?,
        AnyValue::List(Series::new("".into(), &[10 as IdxSize, 10 as IdxSize]))
    );

    let out = df
        .clone()
        .lazy()
        .group_by([col("name")])
        .agg([when(col("value").sum().eq(lit(3)))
            .then(col("value").rank(Default::default(), None))
            .otherwise(Null {}.lit())])
        .sort(["name"], Default::default())
        .collect()?;

    let out = out.column("value")?;
    assert!(matches!(out.get(0)?, AnyValue::List(_)));
    assert!(matches!(out.get(1)?, AnyValue::List(_)));

    // swapped branch
    let out = df
        .lazy()
        .group_by([col("name")])
        .agg([when(col("value").sum().eq(lit(3)))
            .then(Null {}.lit().alias("value"))
            .otherwise(col("value").rank(Default::default(), None))])
        .sort(["name"], Default::default())
        .collect()?;

    let out = out.column("value")?;
    assert!(matches!(out.get(1)?, AnyValue::List(_)));
    assert!(matches!(out.get(0)?, AnyValue::List(_)));

    Ok(())
}

#[test]
fn test_binary_group_consistency() -> PolarsResult<()> {
    let lf = df![
        "name" => ["a", "b", "c", "d"],
        "category" => [1, 2, 3, 4],
        "score" => [3, 5, 1, 2],
    ]?
    .lazy();

    let out = lf
        .group_by([col("category")])
        .agg([col("name").filter(col("score").eq(col("score").max()))])
        .sort(["category"], Default::default())
        .collect()?;
    let out = out.column("name")?;

    assert_eq!(out.dtype(), &DataType::List(Box::new(DataType::String)));
    assert_eq!(
        out.explode(ExplodeOptions {
            empty_as_null: true,
            keep_nulls: true
        })?
        .str()?
        .no_null_iter()
        .collect::<Vec<_>>(),
        &["a", "b", "c", "d"]
    );

    Ok(())
}

fn assert_lazy_and_streaming_eq(mut lf: LazyFrame, expected: &DataFrame) -> PolarsResult<()> {
    let schema = lf.collect_schema()?;
    assert_eq!(schema.as_ref(), expected.schema().as_ref());

    let in_mem = lf.clone().collect()?;
    assert_eq!(in_mem.schema(), expected.schema());
    assert!(in_mem.equals_missing(expected));

    #[cfg(feature = "streaming")]
    {
        let streaming = lf.with_streaming(true).collect()?;
        assert_eq!(streaming.schema(), expected.schema());
        assert!(streaming.equals_missing(expected));
    }

    Ok(())
}

#[test]
fn test_when_then_dyn_int_unsigned_arithmetic_30674() -> PolarsResult<()> {
    let df = df![
        "side" => ["buy", "sell"],
        "qty" => [10u32, 4u32],
    ]?;

    let pos_neg = when(col("side").eq(lit("buy"))).then(1).otherwise(-1);
    let neg_pos = when(col("side").eq(lit("buy"))).then(-1).otherwise(1);

    let q = df.clone().lazy().select([
        (pos_neg.clone() * col("qty")).alias("signed_qty"),
        (neg_pos * col("qty")).alias("rev_signed_qty"),
        (pos_neg.clone() + col("qty")).alias("added"),
        (pos_neg.clone() - col("qty")).alias("subtracted"),
        pos_neg.clone().lt(col("qty")).alias("lt_cmp"),
        pos_neg.clone().eq(col("qty")).alias("eq_cmp"),
    ]);
    let expected = df![
        "signed_qty" => [10i64, -4i64],
        "rev_signed_qty" => [-10i64, 4i64],
        "added" => [11i64, 3i64],
        "subtracted" => [-9i64, -5i64],
        "lt_cmp" => [true, true],
        "eq_cmp" => [false, false],
    ]?;
    assert_lazy_and_streaming_eq(q, &expected)?;

    let q_sum = df
        .clone()
        .lazy()
        .select([(pos_neg.clone() * col("qty")).sum().alias("net_qty")]);
    assert_lazy_and_streaming_eq(q_sum, &df!["net_qty" => [6i64]]?)?;

    assert_lazy_and_streaming_eq(
        df.clone().lazy().select([pos_neg.clone().alias("sign")]),
        &df!["sign" => [1i32, -1i32]]?,
    )?;
    let mut q_wide = df.lazy().select([when(col("side").eq(lit("buy")))
        .then(1)
        .otherwise(10_i64.pow(12))
        .alias("wide")]);
    assert_eq!(q_wide.collect_schema()?.get("wide"), Some(&DataType::Int64));

    let df_nulls = df![
        "side" => [Some("buy"), Some("sell"), None, Some("buy"), Some("sell")],
        "qty" => [Some(10u32), Some(4u32), Some(5u32), None, Some(0u32)],
    ]?;
    assert_lazy_and_streaming_eq(
        df_nulls
            .lazy()
            .select([(pos_neg * col("qty")).alias("signed_qty")]),
        &df!["signed_qty" => [Some(10i64), Some(-4i64), Some(-5i64), None, Some(0i64)]]?,
    )?;

    Ok(())
}

#[test]
#[cfg(all(
    feature = "dtype-u8",
    feature = "dtype-u16",
    feature = "dtype-i8",
    feature = "dtype-i16",
    feature = "dtype-u128",
    feature = "dtype-i128"
))]
fn test_when_then_dyn_int_width_matrix_30674() -> PolarsResult<()> {
    let base = df![
        "side" => ["buy", "sell"],
        "qty" => [10i64, 4i64],
    ]?;
    let pos_neg = when(col("side").eq(lit("buy"))).then(1).otherwise(-1);
    let pos_only = when(col("side").eq(lit("buy"))).then(1).otherwise(2);

    for (input_dtype, expected_dtype) in [
        (DataType::UInt8, DataType::Int16),
        (DataType::UInt16, DataType::Int32),
        (DataType::UInt32, DataType::Int64),
        (DataType::UInt64, DataType::Int64),
        (DataType::UInt128, DataType::Int128),
    ] {
        let mut df = base.clone();
        df.with_column(df.column("qty")?.cast(&input_dtype)?)?;
        let q = df
            .lazy()
            .select([(pos_neg.clone() * col("qty")).alias("out")]);
        let expected_col = df!["out" => [10i64, -4i64]]?
            .column("out")?
            .cast(&expected_dtype)?;
        let expected = DataFrame::new(2, vec![expected_col])?;
        assert_lazy_and_streaming_eq(q, &expected)?;
    }

    for unsigned_dtype in [
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::UInt128,
    ] {
        let mut df = base.clone();
        df.with_column(df.column("qty")?.cast(&unsigned_dtype)?)?;
        let q = df
            .lazy()
            .select([(pos_only.clone() * col("qty")).alias("out")]);
        let expected_col = df!["out" => [10i64, 8i64]]?
            .column("out")?
            .cast(&unsigned_dtype)?;
        let expected = DataFrame::new(2, vec![expected_col])?;
        assert_lazy_and_streaming_eq(q, &expected)?;
    }

    for signed_dtype in [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::Int128,
    ] {
        let mut df = base.clone();
        df.with_column(df.column("qty")?.cast(&signed_dtype)?)?;
        let q = df
            .lazy()
            .select([(pos_neg.clone() * col("qty")).alias("out")]);
        let expected_col = df!["out" => [10i64, -4i64]]?
            .column("out")?
            .cast(&signed_dtype)?;
        let expected = DataFrame::new(2, vec![expected_col])?;
        assert_lazy_and_streaming_eq(q, &expected)?;
    }

    let df_u8 = df![
        "side" => ["buy", "sell"],
        "qty" => [10u8, 4u8],
    ]?;
    assert_lazy_and_streaming_eq(
        df_u8
            .lazy()
            .select([
                (when(col("side").eq(lit("buy"))).then(-1).otherwise(-200) * col("qty"))
                    .alias("out"),
            ]),
        &df!["out" => [-10i16, -800i16]]?,
    )?;

    let df_i8 = df![
        "side" => ["buy", "sell"],
        "x" => [1i8, 1i8],
    ]?;
    assert_lazy_and_streaming_eq(
        df_i8
            .lazy()
            .select([
                (when(col("side").eq(lit("buy"))).then(128).otherwise(-1) + col("x")).alias("out"),
            ]),
        &df!["out" => [129i16, 0i16]]?,
    )?;

    let df_u64_big = df![
        "side" => ["buy", "sell"],
        "qty" => [(1u64 << 63) + 1, 4u64],
    ]?;
    assert_lazy_and_streaming_eq(
        df_u64_big
            .lazy()
            .select([(pos_neg * col("qty")).alias("out")]),
        &df!["out" => [None, Some(-4i64)]]?,
    )?;

    Ok(())
}

#[test]
fn test_dyn_int_sibling_supertype_paths_30674() -> PolarsResult<()> {
    let df = df![
        "side" => ["buy", "sell"],
        "qty" => [10u32, 4u32],
    ]?;

    let q_list = df
        .clone()
        .lazy()
        .select([concat_list([lit(1), lit(-1), col("qty")])?.alias("out")]);
    let expected_list = df![
        "out" => [
            AnyValue::List(Series::new("".into(), [1i64, -1, 10])),
            AnyValue::List(Series::new("".into(), [1i64, -1, 4])),
        ]
    ]?;
    assert_lazy_and_streaming_eq(q_list, &expected_list)?;

    #[cfg(feature = "list_arithmetic")]
    {
        let q_list_mul = df
            .clone()
            .lazy()
            .select([(concat_list([lit(1), lit(-1)])? * col("qty")).alias("out")]);
        let expected_list_mul = df![
            "out" => [
                AnyValue::List(Series::new("".into(), [10i64, -10])),
                AnyValue::List(Series::new("".into(), [4i64, -4])),
            ]
        ]?;
        assert_lazy_and_streaming_eq(q_list_mul, &expected_list_mul)?;
    }

    #[cfg(feature = "coalesce")]
    {
        let branch_with_nulls = when(col("side").eq(lit("buy")))
            .then(1)
            .otherwise(lit(LiteralValue::untyped_null()));
        let q_coalesce = df
            .lazy()
            .select([(coalesce(&[branch_with_nulls, lit(-1)]) * col("qty")).alias("out")]);
        assert_lazy_and_streaming_eq(q_coalesce, &df!["out" => [10i64, -4i64]]?)?;
    }

    Ok(())
}
