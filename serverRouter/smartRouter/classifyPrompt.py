from serverRouter.smartRouter.taskEmbeddingManager import task_manager

def classify_prompt(query):
    """
    Classifies a given prompt and returns a dictionary of similar tasks with their normalized similarity scores.
    
    Args:
        query (str): The prompt to classify
        task_manager: The task manager object with find_similar_tasks method
        loaded_embeddings: The embeddings to use for classification
        
    Returns:
        dict: Dictionary mapping task_id to similarity score (only includes scores > 0)
    """
    
    similar_tasks = task_manager.find_similar_tasks(query)
    if not similar_tasks:
        return {}

    # Normalize similarity scores to the 0-1 range. If every score is identical
    # (score_range == 0) the query matched all candidate tasks equally, so treat
    # them all as fully relevant instead of dividing by zero.
    scores = [score for _, score in similar_tasks]
    min_score = min(scores)
    score_range = max(scores) - min_score

    result = {}
    for task_id, score in similar_tasks:
        normalized = (score - min_score) / score_range if score_range else 1.0
        if normalized >= 0.5:
            result[task_id] = float(round(normalized, 4))

    return result


# Example usage:
if __name__ == "__main__":
    query = "code a python program to calculate the sum of all numbers in a list"
    print(f"\nFinding tasks similar to: '{query}'")
    
    similar_tasks_dict = classify_prompt(query)
    
    print("\nTop similar tasks:")
    for task_id, similarity in similar_tasks_dict.items():
        print(f"- {task_id} (Similarity: {similarity:.4f})")