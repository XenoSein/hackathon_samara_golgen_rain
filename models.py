# coding: utf-8
import os
import warnings
from contextlib import contextmanager
from operator import or_
from typing import List

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, text, inspect, Text, Float, Index, func, \
    not_
from sqlalchemy import create_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.sql import Insert

Base = declarative_base()
metadata = Base.metadata


def universal_sqla_stringify(self):
    pks = [f"{column.name}={getattr(self, column.name)}" for column in
           list(filter(lambda column: column.primary_key, inspect(self.__class__).columns))]
    return f"{self.__class__.__name__}({', '.join(pks)})"


def universal_sqla_repr(self):
    return self.__str__()


Base.__str__ = universal_sqla_stringify
Base.__repr__ = universal_sqla_repr


@compiles(Insert, "postgresql")
def postgresql_on_conflict_do_nothing(insert, compiler, **kw):
    """ Обработчик на все insert, отменяет ошибку duplicate key. """
    statement = compiler.visit_insert(insert, **kw)
    returning_position = statement.find("RETURNING")
    if returning_position >= 0:
        return statement[:returning_position] + "ON CONFLICT DO NOTHING " + statement[returning_position:]
    else:
        return statement + " ON CONFLICT DO NOTHING"


class Patient(Base):
    __tablename__ = 'patient'
    __table_args__ = (
        Index('patient_selected_result_predicted_result_index', 'selected_result', 'predicted_result'),
    )

    id = Column(Integer, primary_key=True, unique=True)
    anamnesis = Column(Text)
    probability = Column(Float)
    task_id = Column(ForeignKey('task.id', ondelete='CASCADE', onupdate='CASCADE'))
    selected_result = Column(Integer)
    predicted_result = Column(Integer)
    factors = Column(Text)

    task = relationship('Task')

    def to_json(self, session):
        return {
            'id': self.id,
            'anamnesis': self.anamnesis,
            'probability': self.probability,
            'selectedResult': self.selected_result,
            'predictedResult': self.predicted_result,
            'isDangerous': bool(self.predicted_result or self.selected_result),
            'factors': self.factors
        }


class Task(Base):
    __tablename__ = 'task'
    id = Column(Integer, primary_key=True, unique=True)
    time = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    status = Column(String(64))
    type = Column(String(64), index=True)

    def to_json(self, session):
        dangerous_count = session.query(func.count(Patient.id)).filter(
            Patient.task_id == self.id,
            or_(Patient.selected_result == 1, Patient.predicted_result == 1)
        ).scalar()
        not_dangerous_count = session.query(func.count(Patient.id)).filter(
            Patient.task_id == self.id,
            not_(or_(Patient.selected_result == 1, Patient.predicted_result == 1))
        ).scalar()
        return {
            'id': self.id,
            'time': self.time.strftime('%d.%m.%Y %H:%M:%S'),
            'status': self.status,
            'type': self.type,
            'dangerous': dangerous_count,
            'notDangerous': not_dangerous_count,
        }


engine = create_engine(
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@postgres.service:5432/"
    f"{os.environ['POSTGRES_DB']}", pool_pre_ping=True, pool_size=32, max_overflow=64
)
db_session = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def SessionManager():
    db = SessionLocal()
    try:
        yield db
    except:
        warnings.warn("auto-rollbacking")
        db.rollback()
        raise
    finally:
        db.close()
