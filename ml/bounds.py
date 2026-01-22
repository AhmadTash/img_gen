
def clamp_params(params: dict) -> dict:
    """
    Clamps prediction values to safe ranges defined in app/main.py.
    """
    clamped = {}
    
    # paint_thickness [1, 200]
    if "paint_thickness" in params:
        val = int(round(params["paint_thickness"]))
        clamped["paint_thickness"] = max(1, min(200, val))

    # messiness [0.0, 1.0]
    if "messiness" in params:
        clamped["messiness"] = max(0.0, min(1.0, float(params["messiness"])))

    # text_wobble [0.0, 1.0]
    if "text_wobble" in params:
        clamped["text_wobble"] = max(0.0, min(1.0, float(params["text_wobble"])))

    # shadow_opacity [0.0, 1.0] (mapped from paint_opacity if needed, or direct)
    if "shadow_opacity" in params:
        clamped["shadow_opacity"] = max(0.0, min(1.0, float(params["shadow_opacity"])))

    # blur_mix [0.0, 1.0] (mapped from grain if needed)
    if "blur_mix" in params:
        clamped["blur_mix"] = max(0.0, min(1.0, float(params["blur_mix"])))
        
    return clamped
