from fastapi import FastAPI

from routers import image, model
from price_prediction.models import CVTargetEncoder as _LegacyCVTargetEncoder
from price_prediction.models import KNNLocalPrice as _LegacyKNNLocalPrice

import sys, types


_main_mod = sys.modules.get("__main__")
if _main_mod is None:
    _main_mod = types.ModuleType("__main__")
    sys.modules["__main__"] = _main_mod

setattr(_main_mod, "CVTargetEncoder", _LegacyCVTargetEncoder)
setattr(_main_mod, "KNNLocalPrice", _LegacyKNNLocalPrice)

# ---------------- App ----------------
app = FastAPI(title="HEStimate API")
app.include_router(image.router)
app.include_router(model.router)


@app.get("/")
async def root():
    return {"message": "Welcome to the HEStimate API, please check /docs for the doc"}
 
