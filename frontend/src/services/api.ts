import type {
  PredictionResponse,
  PredictionHistoryResponse,
  SingleHistoryResponse,
} from '../types/prediction'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export async function analyzeDocument(
  text: string
): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text }),
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(
      data.detail || 'Unable to analyse document. Please try again.'
    )
  }

  return data as PredictionResponse
}

export async function getPredictionHistory(
  limit = 10,
  offset = 0
): Promise<PredictionHistoryResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/history?limit=${limit}&offset=${offset}`
  )

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.detail || 'Unable to load prediction history.')
  }

  return data as PredictionHistoryResponse
}

export async function getPredictionById(
  historyId: number
): Promise<SingleHistoryResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/history/${historyId}`
  )

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.detail || 'Unable to load this prediction.')
  }

  return data as SingleHistoryResponse
}

import type {
  TopicsResponse,
  SingleTopicResponse
} from "../types/prediction";

export const getTopics = async (
  model: string,
  search: string = "",
  includeOutliers: boolean = true
): Promise<TopicsResponse> => {
  const params = new URLSearchParams();

  params.append("model", model);
  params.append("include_outliers", String(includeOutliers));

  if (search.trim()) {
    params.append("search", search.trim());
  }

  const response = await fetch(
    `${API_BASE_URL}/api/v1/topics/?${params.toString()}`
  );

  if (!response.ok) {
    throw new Error("Failed to fetch topics from server.");
  }

  return response.json();
};

export const getTopicById = async (
  model: string,
  topicId: number
): Promise<SingleTopicResponse> => {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/topics/${model}/${topicId}`
  );

  if (!response.ok) {
    throw new Error("Failed to fetch topic details.");
  }

  return response.json();
};