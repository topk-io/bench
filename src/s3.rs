use std::env;
use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Instant,
};

use aws_config::Region;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::{config::Credentials, Client, Config};
use tokio::fs::File as TokioFile;
use tracing::{debug, info};

pub(crate) fn new_client() -> anyhow::Result<Client> {
    let creds = Credentials::new(
        env::var("AWS_ACCESS_KEY_ID")?,
        env::var("AWS_SECRET_ACCESS_KEY")?,
        None,
        None,
        "topk-bench",
    );

    let mut builder = Config::builder()
        .region(Region::new(env::var("AWS_REGION")?))
        .credentials_provider(creds)
        .endpoint_url(format!(
            "https://s3.{}.amazonaws.com",
            env::var("AWS_REGION")?
        ));

    // Disable the following warning: This checksum is a part-level checksum which can't be validated by the Rust SDK. Disable checksum validation for this request to fix this warning. more_info="See https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity.html#large-object-checksums for more information."
    builder.set_request_checksum_calculation(None);

    Ok(Client::from_conf(builder.build()))
}

pub async fn ensure_file(
    path: impl Into<String>,
    out_dir: impl Into<String>,
) -> anyhow::Result<PathBuf> {
    let path = path.into();

    let path = if path.starts_with("s3://") {
        pull_file(path, out_dir).await?
    } else {
        PathBuf::from(path)
    };

    Ok(path)
}

pub async fn upload_file(bucket: &str, key: &str, file: PathBuf) -> anyhow::Result<()> {
    let s3 = new_client()?;

    let body = ByteStream::from_path(file).await?;

    let response = s3
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(body)
        .send()
        .await?;
    debug!(?response, "File written to S3");

    Ok(())
}

pub async fn open_file(
    path: impl Into<String>,
    cache_dir: impl Into<String>,
) -> anyhow::Result<File> {
    let path = ensure_file(path, cache_dir).await?;

    let file = TokioFile::open(path).await?;
    let file = file
        .try_into_std()
        .expect("Failed to convert tokio::fs::File to std::fs::File");

    Ok(file)
}

async fn pull_file(url: String, out_dir: impl Into<String>) -> anyhow::Result<PathBuf> {
    let cache_dir = out_dir.into();

    let url = url.replace("s3://", "");
    let (bucket, key) = url.split_once("/").unwrap();
    debug!(?bucket, ?key, "Pulling dataset");

    // Ensure the tmp directory exists first
    let dir = Path::new(&cache_dir);
    if !dir.exists() {
        std::fs::create_dir_all(dir)?;
    }

    let out = format!("{cache_dir}/{key}");
    if Path::new(&out).exists() {
        debug!(?out, "Dataset already downloaded");
        return Ok(PathBuf::from(out));
    }

    info!(?bucket, ?key, "Downloading dataset");

    // Download dataset
    let s3 = new_client()?;

    let start = Instant::now();
    let resp = s3.get_object().bucket(bucket).key(key).send().await?;
    let mut data = resp.body.into_async_read();
    // Ensure the directory exists
    std::fs::create_dir_all(Path::new(&out).parent().unwrap())?;
    let mut file = tokio::fs::File::create(&out).await?;
    tokio::io::copy(&mut data, &mut file).await?;
    let duration = start.elapsed();

    info!(?out, ?duration, "Dataset downloaded");

    Ok(PathBuf::from(out))
}
