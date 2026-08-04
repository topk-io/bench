use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::datatypes::DataType;
use arrow::datatypes::Field;
use arrow::datatypes::Schema;
use arrow_array::ArrayRef;
use arrow_array::Float64Array;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_array::TimestampMicrosecondArray;
use arrow_schema::TimeUnit;
use parquet::arrow::ArrowWriter;
use tracing::info;

use crate::s3::upload_file;
use crate::telemetry::metrics::flush_metrics;
use crate::telemetry::metrics::Metric;

pub async fn export(path: &str) -> anyhow::Result<()> {
    let metrics = flush_metrics().await;

    if path.starts_with("s3://") {
        let (_, bucket_uri) = path.split_once("://").expect("Invalid S3 path");
        let (bucket, key) = bucket_uri.split_once("/").expect("Invalid S3 path");

        let tmp_dir = tempfile::tempdir()?;
        let tmp_file = tmp_dir
            .path()
            .join(format!("{}.parquet", uuid::Uuid::new_v4()));

        write_to_file(metrics, tmp_file.clone())?;
        write_to_s3(bucket, key, tmp_file).await?;
        info!("Metrics written to s3://{bucket}/{key}");
    } else {
        write_to_file(metrics, PathBuf::from(path))?;
        info!("Metrics written to {path}");
    }

    Ok(())
}

async fn write_to_s3(bucket: &str, key: &str, path: PathBuf) -> anyhow::Result<()> {
    upload_file(bucket, key, path).await
}

fn write_to_file(metrics: Vec<Metric>, path: PathBuf) -> anyhow::Result<()> {
    let file = std::fs::File::create(path)?;

    // Find all unique label keys (union of all label sets in the metrics)
    let label_keys = {
        let mut set = BTreeSet::new();
        for metric in &metrics {
            for k in metric.metadata.keys() {
                set.insert(k.clone());
            }
        }
        set.into_iter().collect::<Vec<String>>()
    };

    // Compose the schema: timestamp, metric, value, ...label_keys
    let schema = {
        let mut fields = vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("metric", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
        ];
        for key in &label_keys {
            fields.push(Field::new(key, DataType::Utf8, false));
        }
        Arc::new(Schema::new(fields))
    };

    // Collect data into column vectors, in schema order
    let mut timestamps = Vec::with_capacity(metrics.len());
    let mut names = Vec::with_capacity(metrics.len());
    let mut values = Vec::with_capacity(metrics.len());
    let mut labels_vecs: Vec<Vec<String>> = (0..label_keys.len())
        .map(|_| Vec::with_capacity(metrics.len()))
        .collect();

    for metric in metrics {
        timestamps.push(metric.timestamp.timestamp_micros());
        names.push(metric.name);
        values.push(metric.value);
        for (i, key) in label_keys.iter().enumerate() {
            labels_vecs[i].push(metric.metadata.get(key).cloned().unwrap_or_default());
        }
    }

    // Build Arrow arrays in order: ts, metric, value, ...labels
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(TimestampMicrosecondArray::from(timestamps)) as ArrayRef,
        Arc::new(StringArray::from(names)) as ArrayRef,
        Arc::new(Float64Array::from(values)) as ArrayRef,
    ];
    for values in labels_vecs {
        arrays.push(Arc::new(StringArray::from(values)) as ArrayRef);
    }

    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}
