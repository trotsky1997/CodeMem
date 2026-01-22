#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTR prediction model for Phase 6.1.

Implements Logistic Regression CTR model with:
- Feature extraction
- Heuristic weight initialization
- Training with gradient descent
- Prediction
"""

import numpy as np
from typing import List, Dict, Any, Optional
from feature_extractor import FeatureExtractor


class LogisticRegressionCTR:
    """
    Logistic Regression CTR prediction model.

    Features:
    - Simple, fast, interpretable
    - Works with small data
    - Heuristic weight initialization for cold start
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.feature_names = self.feature_extractor.get_feature_names()
        self.weights = None
        self.bias = 0.0

    def extract_features(
        self,
        query: str,
        conversation: Dict[str, Any],
        user_history: Dict[str, Any],
        context: Dict[str, Any],
        position: int
    ) -> np.ndarray:
        """
        Extract features as numpy array.

        Args:
            query: Query string
            conversation: Conversation dict
            user_history: User history dict
            context: Context dict
            position: Result position

        Returns:
            Feature vector
        """
        features_dict = self.feature_extractor.extract_all(
            query, conversation, user_history, context, position
        )

        # Convert to numpy array (sorted by feature names)
        feature_vector = np.array([features_dict[name] for name in self.feature_names])

        return feature_vector

    def predict(self, features: np.ndarray) -> float:
        """
        Predict CTR.

        Args:
            features: Feature vector

        Returns:
            Predicted CTR (0-1)
        """
        if self.weights is None:
            self._initialize_weights()

        # Logistic function: P = 1 / (1 + exp(-w·x - b))
        logit = np.dot(self.weights, features) + self.bias
        ctr = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))  # Clip to avoid overflow

        return float(ctr)

    def predict_batch(self, features_list: List[np.ndarray]) -> List[float]:
        """
        Predict CTR for multiple samples.

        Args:
            features_list: List of feature vectors

        Returns:
            List of predicted CTRs
        """
        return [self.predict(features) for features in features_list]

    def train(self, training_data: List[Dict[str, Any]], epochs: int = 100, learning_rate: float = 0.01):
        """
        Train model with gradient descent.

        Args:
            training_data: List of training samples with 'features' and 'label'
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        if not training_data:
            return

        # Extract features and labels
        X = []
        y = []
        weights_list = []

        for sample in training_data:
            if 'features' in sample:
                features = sample['features']
            else:
                # Extract features on the fly
                features = self.extract_features(
                    sample['query'],
                    sample.get('conversation', {}),
                    sample.get('user_history', {}),
                    sample.get('context', {}),
                    sample.get('position', 1)
                )

            X.append(features)
            y.append(sample['label'])
            weights_list.append(sample.get('weight', 1.0))

        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights_list)

        # Initialize weights if needed
        if self.weights is None:
            self._initialize_weights()

        # Gradient descent
        for epoch in range(epochs):
            # Forward pass
            logits = np.dot(X, self.weights) + self.bias
            logits = np.clip(logits, -500, 500)  # Prevent overflow
            predictions = 1.0 / (1.0 + np.exp(-logits))

            # Calculate weighted loss
            errors = predictions - y
            weighted_errors = errors * weights

            # Gradients
            grad_w = np.dot(X.T, weighted_errors) / len(y)
            grad_b = np.mean(weighted_errors)

            # Update weights
            self.weights -= learning_rate * grad_w
            self.bias -= learning_rate * grad_b

            # Log progress
            if epoch % 20 == 0:
                loss = -np.mean(
                    weights * (y * np.log(predictions + 1e-10) + (1 - y) * np.log(1 - predictions + 1e-10))
                )
                print(f"  Epoch {epoch:3d}, Loss: {loss:.4f}")

    def _initialize_weights(self):
        """
        Initialize weights with heuristic values.

        Based on domain knowledge about feature importance.
        """
        weights_dict = {}

        # Match features (most important)
        weights_dict['bm25_score'] = 2.0
        weights_dict['keyword_overlap'] = 1.5
        weights_dict['tech_keyword_overlap'] = 1.5
        weights_dict['exact_match_count'] = 1.0
        weights_dict['query_in_title'] = 1.0

        # ConversationRank
        weights_dict['conversation_rank'] = 1.5

        # Conversation quality
        weights_dict['has_solution'] = 1.0
        weights_dict['has_code_block'] = 0.8
        weights_dict['message_count'] = 0.01  # Normalized
        weights_dict['code_block_count'] = 0.3
        weights_dict['has_confirmation'] = 0.5
        weights_dict['turn_count'] = 0.05
        weights_dict['tech_keyword_density'] = 0.5
        weights_dict['unique_tech_keywords'] = 0.1

        # Time features
        weights_dict['is_recent'] = 0.5
        weights_dict['days_ago'] = -0.01  # Negative: older = less relevant

        # Context features
        weights_dict['is_same_session'] = 1.2
        weights_dict['is_recent_session'] = 0.6
        weights_dict['is_follow_up'] = 0.5

        # Position features
        weights_dict['position_bias'] = 0.5
        weights_dict['is_top_1'] = 0.3
        weights_dict['is_top_3'] = 0.2
        weights_dict['is_top_5'] = 0.1

        # User preference features
        weights_dict['user_topic_preference'] = 0.8
        weights_dict['user_prefers_recent'] = 0.3
        weights_dict['prefers_long_conversations'] = 0.2
        weights_dict['prefers_code_heavy'] = 0.3
        weights_dict['has_referenced_before'] = 1.0

        # Pattern clustering features (Phase 4.5 integration)
        weights_dict['in_frequent_query_cluster'] = 0.9  # High weight - user often searches this
        weights_dict['session_type_match'] = 0.7  # Medium-high - matches user's preferred session type
        weights_dict['is_recurring_problem'] = 1.2  # Very high - recurring problems are important
        weights_dict['topic_cluster_match'] = 0.8  # High - matches user's frequent topics

        # Query features
        weights_dict['query_length'] = 0.001
        weights_dict['query_word_count'] = 0.01
        weights_dict['has_code'] = 0.3
        weights_dict['is_how_question'] = 0.2
        weights_dict['is_why_question'] = 0.2
        weights_dict['is_what_question'] = 0.1
        weights_dict['tech_keyword_count'] = 0.1
        weights_dict['has_version_number'] = 0.2
        weights_dict['has_time_expression'] = 0.1

        # Conversation features
        weights_dict['avg_message_length'] = 0.0001

        # Position
        weights_dict['position'] = -0.1  # Negative: higher position = less relevant

        # Default weight for any missing features
        self.weights = np.array([weights_dict.get(name, 0.1) for name in self.feature_names])
        self.bias = 0.0

    def get_feature_importance(self) -> List[tuple]:
        """
        Get feature importance (absolute weight values).

        Returns:
            List of (feature_name, weight) tuples sorted by importance
        """
        if self.weights is None:
            self._initialize_weights()

        importance = [(name, abs(weight)) for name, weight in zip(self.feature_names, self.weights)]
        importance.sort(key=lambda x: x[1], reverse=True)

        return importance


class CTRRanker:
    """
    CTR-based ranker for search results.

    Combines BM25 initial ranking with CTR prediction for re-ranking.
    """

    def __init__(self, model: Optional[LogisticRegressionCTR] = None):
        """
        Initialize ranker.

        Args:
            model: Pre-trained CTR model (optional)
        """
        self.model = model or LogisticRegressionCTR()

    def rank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        user_history: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates by predicted CTR.

        Args:
            query: Query string
            candidates: List of candidate conversations
            user_history: User history dict
            context: Context dict

        Returns:
            Ranked list of candidates with predicted_ctr field
        """
        # Extract features and predict CTR for each candidate
        for position, candidate in enumerate(candidates, 1):
            features = self.model.extract_features(
                query, candidate, user_history, context, position
            )
            candidate['predicted_ctr'] = self.model.predict(features)
            candidate['original_position'] = position

        # Sort by predicted CTR
        candidates.sort(key=lambda x: x['predicted_ctr'], reverse=True)

        return candidates

    def train(self, training_data: List[Dict[str, Any]], epochs: int = 100):
        """
        Train the CTR model.

        Args:
            training_data: Training samples
            epochs: Number of epochs
        """
        print(f"🔄 Training CTR model with {len(training_data)} samples...")
        self.model.train(training_data, epochs=epochs)
        print("✅ Training complete")


def create_ctr_model_from_distance_data(
    events: List[Dict[str, Any]],
    method: str = 'hybrid',
    epochs: int = 100
) -> LogisticRegressionCTR:
    """
    Create and train CTR model from distance-based training data.

    Args:
        events: List of events
        method: Training data generation method
        epochs: Training epochs

    Returns:
        Trained CTR model
    """
    from distance_trainer import generate_distance_training_data
    from conversation_rank import calculate_conversation_ranks

    print("🔄 Generating training data...")

    # Calculate ConversationRanks
    ranks = calculate_conversation_ranks(events)

    # Generate distance-based training data
    training_data = generate_distance_training_data(events, method=method)

    # Attach ConversationRank to training data
    for sample in training_data:
        session_id = sample.get('candidate_session_id')
        if session_id:
            sample['conversation'] = {
                'session_id': session_id,
                'messages': [],  # Simplified
                'rank': ranks.get(session_id, 0.5)
            }

    print(f"✅ Generated {len(training_data)} training samples")

    # Create and train model
    model = LogisticRegressionCTR()
    ranker = CTRRanker(model)
    ranker.train(training_data, epochs=epochs)

    return model
