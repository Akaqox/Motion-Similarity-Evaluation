def rc_loss(rc_anchor, anchor, loss_fn):
    """Calculates the L_rec loss."""
    return loss_fn(rc_anchor, anchor)

def triplet_loss(features, loss_fn):
    """
    Calculates the total triplet loss (L_triplet) for all three encoders.
    
    Args:
        features (dict): A dictionary containing all 9 encoded feature vectors.
        loss_fn (nn.TripletMarginLoss): The triplet loss function.
    """
    
    # L_triplet for Action Encoder (E_m)
    L_triplet_M = loss_fn(
        features["h_a_anchor"], 
        features["h_a_act_pos"], 
        features["h_a_act_neg"]
    )
    
    # L_triplet for Skeleton Encoder (E_s)
    L_triplet_S = loss_fn(
        features["h_s_anchor"], 
        features["h_s_skel_pos"], 
        features["h_s_skel_neg"]
    )
    
    # L_triplet for Viewpoint Encoder (E_v)
    L_triplet_V = loss_fn(
        features["h_v_anchor"], 
        features["h_v_view_pos"], 
        features["h_v_view_neg"]
    )
    
    return L_triplet_M + L_triplet_S + L_triplet_V

def cross_rc_loss(rc_cross, cross_target, loss_fn):
    """Calculates the L_cross loss."""
    return loss_fn(rc_cross, cross_target)