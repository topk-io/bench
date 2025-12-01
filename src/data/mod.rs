mod doc;
pub use doc::parse_from_batch;
pub use doc::Document;

mod query;
pub use query::load_from_path;
pub use query::Query;
