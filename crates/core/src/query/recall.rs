use std::collections::HashSet;

use crate::data::{Document, Query};
use crate::query::QueryConfig;

pub fn calculate_recall(
    results: Vec<Document>,
    query: Query,
    config: &QueryConfig,
) -> anyhow::Result<f32> {
    let actual_doc_ids = results
        .iter()
        .map(|x| Ok(x.id.parse()?))
        .collect::<anyhow::Result<HashSet<u32>>>()?;

    let expected_doc_ids = recall(&query, config)?;
    let found_doc_ids = actual_doc_ids.intersection(&expected_doc_ids).count();

    Ok(found_doc_ids as f32 / expected_doc_ids.len() as f32)
}

fn recall(query: &Query, config: &QueryConfig) -> anyhow::Result<HashSet<u32>> {
    assert!(
        config.top_k <= 100,
        "top_k must be less than or equal to 100"
    );

    let int_filter = config.int_filter.unwrap_or(10000);
    let keyword_filter = config.keyword_filter.clone().unwrap_or("10000".to_string());

    let doc_ids = query
        .recall
        .get(&int_filter)
        .expect("int_filter not found")
        .get(&keyword_filter)
        .expect("keyword_filter not found")
        .clone()
        .into_iter()
        .filter(|x| x.is_positive())
        .map(|x| x as u32)
        .take(config.top_k as usize)
        .collect();

    Ok(doc_ids)
}
