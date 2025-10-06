import numpy as np
from typing import Tuple, List, Union, Literal, Callable
from ..config import LOGGER

def count_nonan(matrix:np.ndarray)->int:
    """Returns the number of non nan in a matrix"""
    return np.count_nonzero(~np.isnan(matrix))

def find_context(canvas:np.ndarray, region:Tuple[int,int,int,int], context_size:int, score_methods:Union[List[Callable[[np.ndarray],float]],Callable[[np.ndarray],float]]=count_nonan, method:Literal["order","mean"]="mean")->Tuple[int,int,int,int]:
    """
    Find the best context(matrix) for a region(matrix) in a large matrix (named canvas).
    The best matrix is choosen by finding the maximum of a score function.
    Args:
        canvas: the large matrix which contains the region and context
        region: region defined by bouding box (square) where each integer is the indice of the corner [x1,y1,x2,y2].
        context_size(int): context size, need to be higher than region size.
        score_methods: score functions
        method: method to combine the score functions.
    Returns:
        context: bouding box of the context
    """

    assert len(canvas.shape) == 2, LOGGER.error("Can't find context because canvas is not 2D (i.e not a matrix).")

    assert region[2]-region[0]!=region[3]-region[1], LOGGER.error("Region need to be a square")
    region_size:int = region[2]-region[0]
    assert context_size > region_size, LOGGER.error("Context size need to higher than the region size.")

    if callable(score_methods):
        score_methods = [score_methods]

    steps = context_size-region_size
    scores = np.zeros((steps,steps,len(score_methods)))

    for j in range(len(steps)):
        for i in range(len(steps)):
            #if(all(scores[k] > 0 for k in scores)):
            #    continue
            x1 = max(region[0] - i, 0)
            y1 = max(region[1] - j, 0)
            x2 = min(x1 + context_size, canvas.shape[0])
            y2 = min(y1 + context_size, canvas.shape[1])

            if x2 - x1 < context_size or y2 - y1 < context_size:
                continue

            context = canvas[x1:x2, y1:y2]
            for k, s_method in enumerate(score_methods):
                try:
                    scores[i, j, k] = s_method(context)
                except Exception as e:
                    LOGGER.error(f"Error in score method {k}: {e}")
                    scores[i, j, k] = -np.inf
    
    if method == "mean":
        combined_score = np.nanmean(scores, axis=2)
    else:  # "order"
        # Normalize scores and average ranks
        ranks = np.argsort(np.argsort(scores, axis=None)).reshape(scores.shape)
        combined_score = np.mean(ranks, axis=2)

    best_idx = np.unravel_index(np.nanargmax(combined_score), combined_score.shape)
    best_i, best_j = best_idx

    context_x1 = max(region[0] - best_i, 0)
    context_y1 = max(region[1] - best_j, 0)
    context_x2 = min(context_x1 + context_size, canvas.shape[0])
    context_y2 = min(context_y1 + context_size, canvas.shape[1])

    return context_x1, context_y1, context_x2, context_y2