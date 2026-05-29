from __future__ import annotations

from dataclasses import dataclass

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class PipelineResult:
    accuracy: float
    report: str


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(n_estimators=200, random_state=42),
            ),
        ]
    )


def run_pipeline(test_size: float = 0.2, random_state: int = 42) -> PipelineResult:
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=random_state,
        stratify=data.target,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=data.target_names)

    return PipelineResult(accuracy=accuracy, report=report)


def main() -> None:
    result = run_pipeline()
    print(f"Wine quality classification accuracy: {result.accuracy:.4f}")
    print("Classification report:")
    print(result.report)


if __name__ == "__main__":
    main()
