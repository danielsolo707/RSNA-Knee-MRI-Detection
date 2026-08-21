_a5_sub = pd.read_csv('/kaggle/working/submission.csv',
                      dtype={'StudyInstanceUID': str})
assert _a5_sub.columns.tolist()[1:] == A5_LABELS, 'submission schema drift'
if A5_W > 0:
    _a5_ours = np.stack([A5_PREDS[_u]
                         for _u in _a5_sub['StudyInstanceUID'].astype(str)])
    _a5_base_rank = _a5_sub[A5_LABELS].rank(method='average', pct=True)
    _a5_ours_rank = pd.DataFrame(_a5_ours, columns=A5_LABELS,
                                 index=_a5_sub.index).rank(method='average', pct=True)
    _a5_sub[A5_LABELS] = (1.0 - A5_W) * _a5_base_rank + A5_W * _a5_ours_rank
    assert np.isfinite(_a5_sub[A5_LABELS].to_numpy()).all()
    _a5_sub.to_csv('/kaggle/working/submission.csv', index=False)