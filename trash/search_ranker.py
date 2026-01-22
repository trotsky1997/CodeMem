#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTR-based search ranking integration for Phase 6.1.

Integrates CTR model with existing BM25 search.
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from conversation_rank import calculate_conversation_ranks
from distance_trainer import generate_distance_training_data
from ctr_model import LogisticRegressionCTR, CTRRanker


class SearchRanker:
    """
    Search ranker with CTR model.

    Workflow:
    1. BM25 initial ranking (existing)
    2. Calculate ConversationRank for all sessions
    3. Extract features for each result
    4. Predict CTR and re-rank
    """

    def __init__(self, ctr_model: Optional[LogisticRegressionCTR] = None):
        """
        Initialize search ranker.

        Args:
            ctr_model: Pre-trained CTR model (optional)
        """
        self.ctr_model = ctr_model or LogisticRegressionCTR()
        self.ctr_ranker = CTRRanker(self.ctr_model)
        self.conversation_ranks: Dict[str, float] = {}

    async def initialize_from_events(self, events: List[Dict[str, Any]], method: str = 'session_based'):
        """
        Initialize CTR model from historical events.

        Args:
            events: List of event dictionaries
            method: Training data generation method
        """
        print("🔄 Initializing CTR model from historical data...")

        # Calculate ConversationRanks
        print("  📊 Calculating ConversationRanks...")
        self.conversation_ranks = calculate_conversation_ranks(events)
        print(f"  ✅ Calculated ranks for {len(self.conversation_ranks)} sessions")

        # Generate training data
        print(f"  📝 Generating training data (method: {method})...")
        training_data = generate_distance_training_data(events, method=method)
        print(f"  ✅ Generated {len(training_data)} training samples")

        # Attach ConversationRank to training data
        for sample in training_data:
            session_id = sample.get('candidate_session_id')
            if session_id:
                sample['conversation'] = {
                    'session_id': session_id,
                    'messages': [],  # Simplified for training
                    'rank': self.conversation_ranks.get(session_id, 0.5),
                    'bm25_score': 0.5  # Will be filled during actual search
                }

        # Train model
        print("  🔄 Training CTR model...")
        self.ctr_model.train(training_data, epochs=50, learning_rate=0.01)
        print("  ✅ CTR model trained")

        print("✅ Search ranker initialized")

    def rerank_results(
        self,
        query: str,
        bm25_results: List[Dict[str, Any]],
        user_history: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank BM25 results using CTR model.

        Args:
            query: Query string
            bm25_results: BM25 search results
            user_history: User history dict
            context: Context dict

        Returns:
            Re-ranked results with predicted_ctr field
        """
        if not bm25_results:
            return []

        # Prepare candidates with ConversationRank
        candidates = []
        for result in bm25_results:
            session_id = result.get('session_id', '')

            # Create conversation dict
            conversation = {
                'session_id': session_id,
                'timestamp': result.get('timestamp', ''),
                'rank': self.conversation_ranks.get(session_id, 0.5),
                'messages': [
                    {
                        'role': result.get('role', ''),
                        'text': result.get('text', '')
                    }
                ],
                'bm25_score': result.get('score', 0.0)
            }

            # Add to candidates
            candidate = result.copy()
            candidate['conversation'] = conversation
            candidates.append(candidate)

        # Use CTR ranker to re-rank
        ranked = self.ctr_ranker.rank(
            query=query,
            candidates=candidates,
            user_history=user_history or {},
            context=context or {}
        )

        # Remove internal fields
        for result in ranked:
            if 'conversation' in result:
                del result['conversation']

        return ranked

    async def search_and_rank(
        self,
        query: str,
        bm25_search_func,
        limit: int = 20,
        source: str = "both",
        user_history: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        use_ctr: bool = True
    ) -> Dict[str, Any]:
        """
        Perform BM25 search and re-rank with CTR.

        Args:
            query: Query string
            bm25_search_func: BM25 search function (async)
            limit: Result limit
            source: Search source
            user_history: User history
            context: Context
            use_ctr: Whether to use CTR re-ranking

        Returns:
            Search results with CTR ranking
        """
        # Perform BM25 search
        bm25_results = await bm25_search_func(query, limit=limit * 2, source=source)  # Get more for re-ranking

        if not use_ctr or not bm25_results.get('results'):
            return bm25_results

        # Re-rank with CTR
        reranked = self.rerank_results(
            query=query,
            bm25_results=bm25_results['results'],
            user_history=user_history,
            context=context
        )

        # Limit results
        reranked = reranked[:limit]

        # Update result
        result = bm25_results.copy()
        result['results'] = reranked
        result['count'] = len(reranked)
        result['ranking_method'] = 'ctr'

        return result


# Global search ranker instance
_search_ranker: Optional[SearchRanker] = None
_ranker_lock = asyncio.Lock()


async def get_search_ranker() -> SearchRanker:
    """
    Get or create global search ranker.

    Returns:
        SearchRanker instance
    """
    global _search_ranker

    async with _ranker_lock:
        if _search_ranker is None:
            _search_ranker = SearchRanker()

        return _search_ranker


async def initialize_search_ranker(events: List[Dict[str, Any]], method: str = 'session_based'):
    """
    Initialize global search ranker from events.

    Args:
        events: List of event dictionaries
        method: Training data generation method
    """
    ranker = await get_search_ranker()
    await ranker.initialize_from_events(events, method=method)


async def search_with_ctr_ranking(
    query: str,
    bm25_search_func,
    limit: int = 20,
    source: str = "both",
    user_history: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    use_ctr: bool = True
) -> Dict[str, Any]:
    """
    Perform search with CTR ranking.

    Args:
        query: Query string
        bm25_search_func: BM25 search function
        limit: Result limit
        source: Search source
        user_history: User history
        context: Context
        use_ctr: Whether to use CTR ranking

    Returns:
        Search results
    """
    ranker = await get_search_ranker()

    return await ranker.search_and_rank(
        query=query,
        bm25_search_func=bm25_search_func,
        limit=limit,
        source=source,
        user_history=user_history,
        context=context,
        use_ctr=use_ctr
    )
