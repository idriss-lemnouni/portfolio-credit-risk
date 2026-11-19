import numpy as np
from streamlit.testing.v1 import AppTest

from study import DEMOGRAPHIC, HERE, TARGET, features, load_cohort, split, summary


def test_real_split_is_disjoint_and_stratified():
    y = load_cohort()[TARGET].to_numpy()
    parts = split(y)
    assert [len(x) for x in parts] == [18000, 6000, 6000]
    assert len(np.unique(np.concatenate(parts))) == 30000
    assert all(abs(y[x].mean() - y.mean()) < 0.001 for x in parts)


def test_input_boundaries_and_group_metrics():
    x = features(load_cohort().head(50), False)
    assert not {"ID", TARGET, *DEMOGRAPHIC}.intersection(x.columns)
    observed = np.array([0, 0, 1, 1])
    result = summary(observed, np.array([0.1, 0.7, 0.4, 0.9]), 0.5)
    assert result["true_positive_rate"] == result["false_positive_rate"] == result["review_rate"] == 0.5


def test_dashboard_model_and_segment_selection():
    app = AppTest.from_file(str(HERE / "dashboard.py"), default_timeout=30).run()
    assert not app.exception
    app.selectbox[0].select("logistic")
    app.radio[0].set_value("Age bracket").run()
    assert not app.exception
    assert app.metric[0].value == "0.755"
