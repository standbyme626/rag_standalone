from sqlalchemy import String, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base_class import Base

class DDInterInteraction(Base):
    __tablename__ = "ddinter_interactions"
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    drug_a: Mapped[str] = mapped_column(String, index=True)
    drug_b: Mapped[str] = mapped_column(String, index=True)
    severity: Mapped[str] = mapped_column(String) # High, Moderate, Low, Unknown
    description: Mapped[str] = mapped_column(Text) # The risk description
