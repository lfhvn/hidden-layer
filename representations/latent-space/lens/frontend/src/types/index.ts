/**
 * TypeScript type definitions for Latent Lens
 */

export interface Experiment {
  id: number;
  name: string;
  description: string | null;
  model_name: string;
  layer_name: string;
  layer_index: number;
  input_dim: number;
  hidden_dim: number;
  sparsity_coef: number;
  learning_rate: number;
  num_epochs: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  num_samples: number;
  train_loss: number | null;
  reconstruction_loss: number | null;
  sparsity_loss: number | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface Feature {
  id: number;
  experiment_id: number;
  feature_index: number;
  activation_mean: number;
  activation_max: number;
  activation_std: number;
  sparsity: number;
  top_tokens: string[];
  top_token_scores: number[];
  created_at: string;
}

export interface FeatureLabel {
  id: number;
  feature_id: number;
  label: string;
  description: string | null;
  tags: string[];
  confidence: number;
  created_by: string | null;
  created_at: string;
}

export interface FeatureActivation {
  feature_id: number;
  activation_value: number;
}

export interface TokenActivation {
  token: string;
  token_index: number;
  features: FeatureActivation[];
}

export interface AnalyzeResponse {
  text: string;
  tokens: TokenActivation[];
  top_features: [number, number][];
}

export interface ExperimentCreate {
  name: string;
  description?: string;
  model_name: string;
  layer_name: string;
  layer_index: number;
  input_dim: number;
  hidden_dim?: number;
  sparsity_coef?: number;
  learning_rate?: number;
  num_epochs?: number;
}

export interface LabelCreate {
  label: string;
  description?: string;
  tags?: string[];
  confidence?: number;
  created_by?: string;
}

export interface GroupCreate {
  name: string;
  description?: string;
  feature_ids: number[];
  created_by?: string;
}
