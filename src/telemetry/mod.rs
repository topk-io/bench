mod logs;

mod persist;
pub use persist::export;

mod snapshot;
pub use snapshot::Snapshot;

pub mod metrics;

pub fn install() -> anyhow::Result<()> {
    logs::install()
}
