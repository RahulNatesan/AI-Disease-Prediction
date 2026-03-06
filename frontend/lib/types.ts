export interface AnalysisResult {
  best_model: string;
  best_n: number;
  best_score: number;
  best_f1: number;
  class_dist: Record<string, number>;
  model_names: string[];
  n_list: number[];
  error_rates: number[][];
  results: {
    n_genes: number;
    model: string;
    accuracy: number;
    f1: number;
    precision: number;
    recall: number;
    best_params: string;
  }[];
  predictions: { patient: string; disease: string }[];
  pred_counts: { disease: string; count: number }[];
}

export interface Settings {
  nList: string;        // comma-separated string
  cvFolds: number;
  useGridSearch: boolean;
}
