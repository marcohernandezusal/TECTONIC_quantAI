"""FastAPI backend for TECTONIC_quantAI
Ejecuta con:
    uvicorn fastapi_main:app --reload

"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4, UUID

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

# ---------------------------------------------------------------------------
# Proyecto utilidades ya existentes
# ---------------------------------------------------------------------------
from deployment.predict_model import run_prediction
from deployment.retrain_selftraining import run_selftraining
from pipeline.utils import get_latest_model

###############################################################################
# Config & DB engine                                                          #
###############################################################################

import os

DB_URL_DEFAULT = "sqlite:///./tectonic_ai.db"        # ← usa ruta relativa ‘./’
DATABASE_URL = os.getenv("DATABASE_URL", DB_URL_DEFAULT)

echo_sql = bool(os.getenv("SQL_ECHO", "0") == "1")
engine = create_engine(DATABASE_URL, echo=echo_sql, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


class Base(DeclarativeBase):
    pass

###############################################################################
# ORM Models                                                                  #
###############################################################################

class SessionDB(Base):
    """Sesión lógica de un usuario. Se identifica con UUID4 como texto."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID4
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    uploads: Mapped[List["UploadedData"]] = relationship(back_populates="session")
    predictions: Mapped[List["PredictionJob"]] = relationship(back_populates="session")


class UploadedData(Base):
    __tablename__ = "uploads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id"))
    original_filename: Mapped[str] = mapped_column(String)
    stored_path: Mapped[str] = mapped_column(Text)
    mime_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped[SessionDB] = relationship(back_populates="uploads")


class MLModel(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    model_type: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[Optional[str]] = mapped_column(String)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_path: Mapped[Optional[str]] = mapped_column(Text)
    registered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    predictions: Mapped[List["PredictionJob"]] = relationship(back_populates="model")


class PredictionJob(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id"))
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("models.id"))
    input_upload_id: Mapped[int] = mapped_column(Integer, ForeignKey("uploads.id"))
    output_csv: Mapped[str] = mapped_column(Text)
    output_plot: Mapped[str] = mapped_column(Text)
    log_path: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, default="completed")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    model: Mapped[MLModel] = relationship(back_populates="predictions")
    session: Mapped[SessionDB] = relationship(back_populates="predictions")
    input_upload: Mapped[UploadedData] = relationship()


###############################################################################
# Pydantic Schemas                                                            #
###############################################################################

class SessionOut(BaseModel):
    id: UUID
    created_at: datetime

    class Config:
        orm_mode = True


class ModelRegister(BaseModel):
    name: str = Field(..., example="ridge_v1")
    model_type: str = Field(..., pattern="^(supervised|selftraining)$")
    path: str = Field(...)
    metadata_path: Optional[str] = Field(None)
    version: Optional[str] = Field(None)


class ModelOut(BaseModel):
    id: int
    name: str
    model_type: str
    version: Optional[str]
    path: str
    registered_at: datetime

    class Config:
        orm_mode = True


class UploadedOut(BaseModel):
    id: int
    original_filename: str
    stored_path: str
    mime_type: Optional[str]
    size_bytes: int
    uploaded_at: datetime

    class Config:
        orm_mode = True


class PredictionOut(BaseModel):
    id: int
    session_id: UUID
    model_id: int
    output_csv: str
    output_plot: str
    created_at: datetime
    status: str

    class Config:
        orm_mode = True

###############################################################################
# Helpers                                                                     #
###############################################################################

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_or_create_session(db: Session, session_id: Optional[str]) -> SessionDB:
    """Devuelve la sesión existente o la crea si `session_id` es None."""
    if session_id:
        sess = db.get(SessionDB, session_id)
        if not sess:
            raise HTTPException(404, "Session not found")
        return sess
    # crear nueva
    new_id = str(uuid4())
    new_sess = SessionDB(id=new_id)
    db.add(new_sess)
    db.commit()
    db.refresh(new_sess)
    return new_sess


def _save_upload(file: UploadFile, dest_dir: Path) -> Path:
    dest_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S%f")
    dest = dest_dir / f"{timestamp}_{file.filename}"
    dest.write_bytes(file.file.read())
    return dest

###############################################################################
# FastAPI App                                                                 #
###############################################################################

app = FastAPI(title="TECTONIC quantAI API", version="0.2.0")

from fastapi.staticfiles import StaticFiles
STATIC_DIR = Path(__file__).resolve().parent / "training" / "figures"
print(f"▶ Mounting static dir at: {STATIC_DIR}")          # <- debug helper

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# Create tables at startup (dev only; use Alembic in prod)
Base.metadata.create_all(bind=engine)

###############################################################################
# Utility                                                                     #
###############################################################################

@app.get("/health", tags=["Utility"])
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/sessions", response_model=SessionOut, tags=["Sessions"], status_code=201)
def create_session(db: Session = Depends(get_db)):
    sess = get_or_create_session(db, None)
    return sess

###############################################################################
# Model management                                                            #
###############################################################################

@app.post("/models", response_model=ModelOut, tags=["Models"], status_code=201)
def register_model(model_in: ModelRegister, db: Session = Depends(get_db)):
    db_model = MLModel(**model_in.dict())
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model


@app.get("/models", response_model=List[ModelOut], tags=["Models"])
def list_models(model_type: Optional[str] = Query(None, pattern="^(supervised|selftraining)$"), db: Session = Depends(get_db)):
    q = db.query(MLModel)
    if model_type:
        q = q.filter(MLModel.model_type == model_type)
    return q.order_by(MLModel.registered_at.desc()).all()


@app.get("/models/{model_id}", response_model=ModelOut, tags=["Models"])
def get_model(model_id: int, db: Session = Depends(get_db)):
    mdl = db.get(MLModel, model_id)
    if not mdl:
        raise HTTPException(404, "Model not found")
    return mdl

###############################################################################
# Upload helper endpoint (optional)                                           #
###############################################################################

@app.post("/uploads", response_model=UploadedOut, tags=["Uploads"], status_code=201)
def upload_file(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    db: Session = Depends(get_db),
):
    sess = get_or_create_session(db, x_session_id)
    stored_path = _save_upload(file, Path("uploads"))
    up = UploadedData(
        session_id=sess.id,
        original_filename=file.filename,
        stored_path=str(stored_path),
        mime_type=file.content_type,
        size_bytes=len(stored_path.read_bytes()),
    )
    db.add(up)
    db.commit()
    db.refresh(up)
    return up

###############################################################################
# Prediction + Selftraining                                                   #
###############################################################################

@app.post("/predict", response_model=PredictionOut, tags=["Inference"], status_code=201)
def predict(
    background_tasks: BackgroundTasks,
    data_file: UploadFile = File(...),
    model_type: str = Query("supervised", pattern="^(supervised|selftraining)$"),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    db: Session = Depends(get_db),
):
    session = get_or_create_session(db, x_session_id)
    stored_path = _save_upload(data_file, Path("uploads"))

    # Registrar upload en BD
    upload_rec = UploadedData(
        session_id=session.id,
        original_filename=data_file.filename,
        stored_path=str(stored_path),
        mime_type=data_file.content_type,
        size_bytes=len(stored_path.read_bytes()),
    )
    db.add(upload_rec)
    db.commit()
    db.refresh(upload_rec)

    # Inferencia sincrónica – usar librerías del proyecto
    result = run_prediction(model_type, str(stored_path))
    model_path = get_latest_model(model_type)

    db_model = (
        db.query(MLModel).filter(MLModel.path == model_path).first()
        or MLModel(name=Path(model_path).stem, model_type=model_type, path=model_path)
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)

    pred = PredictionJob(
        session_id=session.id,
        model_id=db_model.id,
        input_upload_id=upload_rec.id,
        output_csv=result["csv"],
        output_plot=result["plot"],
        log_path=result["log"],
        status="completed",
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


@app.post("/selftraining", response_model=PredictionOut, tags=["Inference"], status_code=201)
def selftraining(
    background_tasks: BackgroundTasks,
    data_file: UploadFile = File(...),
    retrain: bool = Query(False),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    db: Session = Depends(get_db),
):
    session = get_or_create_session(db, x_session_id)
    stored_path = _save_upload(data_file, Path("uploads"))

    upload_rec = UploadedData(
        session_id=session.id,
        original_filename=data_file.filename,
        stored_path=str(stored_path),
        mime_type=data_file.content_type,
        size_bytes=len(stored_path.read_bytes()),
    )
    db.add(upload_rec)
    db.commit()
    db.refresh(upload_rec)

    if retrain:
        result = run_selftraining(str(stored_path))
    else:
        result = run_prediction("selftraining", str(stored_path))

    model_path = get_latest_model("selftraining")
    db_model = (
        db.query(MLModel).filter(MLModel.path == model_path).first()
        or MLModel(name=Path(model_path).stem, model_type="selftraining", path=model_path)
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)

    pred = PredictionJob(
        session_id=session.id,
        model_id=db_model.id,
        input_upload_id=upload_rec.id,
        output_csv=result["csv"],
        output_plot=result["plot"],
        log_path=result.get("log"),
        status="completed",
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred

###############################################################################
# Artefacts download                                                          #
###############################################################################

@app.get("/predictions/{pred_id}/csv", response_class=FileResponse, tags=["Artefacts"])
def download_csv(
    pred_id: int,
    x_session_id: str = Header(..., alias="X-Session-ID"),
    db: Session = Depends(get_db),
):
    pred = db.get(PredictionJob, pred_id)
    if not pred or pred.session_id != x_session_id:
        raise HTTPException(404, "Prediction not found for this session")
    return FileResponse(pred.output_csv, filename=Path(pred.output_csv).name)


@app.get("/predictions/{pred_id}/plot", response_class=FileResponse, tags=["Artefacts"])
def download_plot(
    pred_id: int,
    x_session_id: str = Header(..., alias="X-Session-ID"),
    db: Session = Depends(get_db),
):
    pred = db.get(PredictionJob, pred_id)
    if not pred or pred.session_id != x_session_id:
        raise HTTPException(404, "Prediction not found for this session")
    return FileResponse(pred.output_plot, filename=Path(pred.output_plot).name)


###############################################################################
# CLI helper                                                                  #
###############################################################################

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "create_tables":
        Base.metadata.create_all(bind=engine)
        print("✅ Tables created at", DATABASE_URL)
    else:
        import uvicorn

        uvicorn.run("fastapi_main:app", host="0.0.0.0", port=8000, reload=True)