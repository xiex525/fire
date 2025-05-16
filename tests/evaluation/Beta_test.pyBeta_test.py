def beta_test(resid: pd.DataFrame, beta: pd.DataFrame, date, window_size: int, factors: pd.DataFrame) -> str:
    """
    对单因子回归中每个 β 做 t 检验 (H0: β=0)，并直接输出 LaTeX 格式的表格。

    :param resid: 回归残差矩阵，形状 (T, N)，index 是日期，columns 是资产名称
    :param beta: β 系数矩阵，index 是日期，columns 是资产名称
    :param date: 要检验的日期
    :param window_size: 窗口大小
    :param factors: 因子收益矩阵，index 是日期，columns 是因子名称（仅支持单因子）
    :return: LaTeX 表格字符串
    """
    # 提取截面 beta
    beta_vals = beta.loc[date].to_numpy()
    if beta_vals.ndim == 1:
        beta_vals = beta_vals.reshape(-1)

    # 构造残差矩阵 N x T
    rows = []
    for i in range(window_size):
        resid_i = resid[i].loc[date].to_numpy()
        rows.append(resid_i)
    resid_mat = np.stack(rows, axis=0).T  # N x T 矩阵

    # 提取因子收益序列长度 window_size
    fac = factors.loc[:date].tail(window_size).to_numpy().reshape(-1)

    N, T = resid_mat.shape
    K = 1
    df = T - K - 1

    # 因子方差 (无偏估计 ddof=0)
    cov_factor = np.var(fac, ddof=0)
    # 各资产残差方差 (ddof=K)
    sigma = [np.var(resid_mat[i], ddof=K) for i in range(N)]

    # 计算 t 统计量和双侧 p 值
    t_stats = beta_vals * np.sqrt(N * cov_factor) / np.sqrt(sigma)
    p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), df))

    # 组织为 DataFrame
    asset_names = resid.columns.tolist() if hasattr(resid, 'columns') else beta.columns.tolist()
    df_result = pd.DataFrame({
        't 统计量': t_stats,
        'p 值': p_values
    }, index=asset_names)
    df_result.index.name = '资产'

    # 生成 LaTeX 表格
    latex_table = df_result.to_latex(
        float_format="%.4f",
        caption="各资产的 t 统计量和 p 值",
        label="tab:beta_t_test",
        escape=False
    )

    print(latex_table)
    return latex_table
