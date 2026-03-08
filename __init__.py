from .zen_face_detailer import ZenFaceDetailer

NODE_CLASS_MAPPINGS = {
    "ZenFaceDetailer": ZenFaceDetailer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZenFaceDetailer": "🎭 ZenFace Detailer (Clownshark Edition)"
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
