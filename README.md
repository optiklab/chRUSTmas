# chRUSTmasLLM

A Rust implementation from scrarch of a classifier that can identify images of cats and non-cats.

cargo add polars -F lazy

    -F (--features)

    Включает Lazy API Polars — ленивые вычисления (аналог Spark/DataFrames):
    - строите вычислительный граф
    - оптимизатор (query optimizer) делает pushdown фильтров, проекций и т.д.
    - выполнение запускается только при .collect()
    Example:
    ```
        use polars::prelude::*;

        fn main() -> PolarsResult<()> {
            let df = LazyFrame::scan_csv("data.csv", Default::default())?
                .filter(col("size").gt(10))           // не выполняется
                .select([col("name"), col("size")])   // не выполняется
                .collect()?; // запуск оптимизатора и выполнения

            println!("{df}");
            Ok(())
    ```
    Альтернатива - eager approach, то есть выполнение сразу:
    ```
    use polars::prelude::*;

    let df = CsvReader::from_path("data.csv")?
        .finish()?;        // загрузка уже выполнена

    let filtered = df.filter(&col("age").gt(30))?;   // выполняется сразу
    ```
    Почему почти всегда стоит использовать Lazy? Потому что lazy-план знает весь граф операций, а значит может:
    - выкинуть ненужные столбцы до чтения файла
    - пропихнуть фильтры (filter pushdown)
    - объединять выражения
    - устранять дубликаты вычислений
    - оптимизировать join‑ы
    - менять порядок операций для максимальной эффективности

    По сути, включение feature lazy превращает Polars в настоящий query optimizer уровня Spark, DuckDB, DataFusion.

cargo add ndarray -F serde

    Фича serde позволяет сериализовать массивы (Array, Array2, ArrayD, …) в JSON, CBOR и другие форматы; десериализовать их обратно. Без serde → ndarray НЕ умеет сериализовать. Фича serde добавляет impl’ы Serialize/Deserialize для ndarray‑типов.

cargo add rand
cargo install (if just cargo.toml copied)