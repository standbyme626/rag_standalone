from sqlalchemy import String, Integer, Text, Enum
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base_class import Base

class DrugTranslation(Base):
    __tablename__ = "medical_rules_translations"
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    cn_name: Mapped[str] = mapped_column(String, unique=True, index=True)
    en_name: Mapped[str] = mapped_column(String, index=True)
    category: Mapped[str] = mapped_column(String, nullable=True) # e.g. "NSAID", "Antibiotic"

class DrugInteraction(Base):
    """
    Supersedes DDInterInteraction with better metadata.
    Designed to store high-risk rules migrated from JSON/Hardcode.
    """
    __tablename__ = "medical_rules_interactions"
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    drug_a: Mapped[str] = mapped_column(String, index=True)
    drug_b: Mapped[str] = mapped_column(String, index=True)
    severity: Mapped[str] = mapped_column(String) # High, Moderate, Low
    description: Mapped[str] = mapped_column(Text)
    evidence_source: Mapped[str] = mapped_column(String, default="manual_rule") # e.g. "DDInter", "FDA", "Manual"

class SafetyGuardrail(Base):
    __tablename__ = "medical_rules_guardrails"
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    keyword: Mapped[str] = mapped_column(String, unique=True, index=True)
    risk_level: Mapped[str] = mapped_column(String, default="high") # high, medium, low
    action_type: Mapped[str] = mapped_column(String, default="block") # block, flag
