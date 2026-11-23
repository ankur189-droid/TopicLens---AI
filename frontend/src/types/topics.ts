
export interface Topic {
  topic_id: number;
  name: string;
  keywords: string[];
  document_count: number | null;
  is_outlier: boolean;
  representative_docs: string[];
}

export interface TopicsResponse {
  status: string;
  model: string;
  total_topics: number;
  topics: Topic[];
}

export interface TopicDetailResponse {
  status: string;
  model: string;
  topic: Topic;
}

export type TopicModel = "lda" | "nmf" | "bertopic";
