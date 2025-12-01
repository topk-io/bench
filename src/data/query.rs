use std::collections::HashMap;

use arrow::json::LineDelimitedWriter;
use arrow_array::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::s3::open_file;

#[allow(dead_code)]
#[derive(serde::Deserialize, Debug, Clone)]
pub struct Query {
    pub dense: Vec<f32>,
    pub recall:
        HashMap</*int_filter*/ u32, HashMap</*keyword_filter*/ String, /*doc IDs*/ Vec<i64>>>,
}

pub async fn load_from_path(path: &str, cache_dir: &str) -> anyhow::Result<Vec<Query>> {
    let file = open_file(path, cache_dir).await?;

    // Load queries in a blocking task to avoid blocking the async runtime
    let batches = tokio::task::spawn_blocking(move || {
        ParquetRecordBatchReaderBuilder::try_new(file)?
            .build()?
            .collect::<Result<Vec<RecordBatch>, _>>()
    })
    .await??;

    Ok(batches
        .iter()
        .map(|batch| -> anyhow::Result<Vec<Query>> {
            let batch = batch_to_buffer(batch)?;

            // Deserialize each JSON line to PqQuery
            let mut vectors = Vec::new();

            for line in batch.lines() {
                if line.trim().is_empty() {
                    continue;
                }

                vectors.push(serde_json::from_str(line)?);
            }

            Ok(vectors)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect())
}

fn batch_to_buffer(batch: &RecordBatch) -> anyhow::Result<String> {
    let mut buffer = Vec::new();
    let mut writer = LineDelimitedWriter::new(&mut buffer);

    writer.write_batches(&[batch])?;
    writer.finish()?;

    Ok(String::from_utf8(buffer)?)
}
