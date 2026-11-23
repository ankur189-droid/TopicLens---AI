export interface LdaPrediction {
  topic_id: number
  confidence: number
  topic_distribution: number[]
}

export interface NmfPrediction {
  topic_id: number
  confidence: number
  topic_distribution: number[]
}

export interface BertopicPrediction {
  topic_id: number
  topic_name: string
  top_words: string[]
  confidence: number | null
}

export interface PredictionResult {
  lda: LdaPrediction
  nmf: NmfPrediction
  bertopic: BertopicPrediction
}

export interface PredictionResponse {
  status: string
  history_id: number
  predictions: PredictionResult
}

export interface PredictionHistoryRecord {
  id: number
  input_text: string
  lda_prediction: LdaPrediction
  nmf_prediction: NmfPrediction
  bertopic_prediction: BertopicPrediction
  created_at: string
}

export interface PredictionHistoryResponse {
  status: string
  total_records: number
  limit: number
  offset: number
  history: PredictionHistoryRecord[]
}

export interface SingleHistoryResponse {
  status: string
  history: PredictionHistoryRecord
}

export interface Topic {
  topic_id: number;
  topic_name: string;
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

export interface SingleTopicResponse {
  status: string;
  model: string;
  topic: Topic;
}