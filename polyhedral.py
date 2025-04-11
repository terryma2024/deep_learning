import islpy as isl

def main():
    # 定义一个迭代空间：for i in [0, N), for j in [0, M)
    ctx = isl.DEFAULT_CONTEXT
    N, M = 64, 32

    # 创建一个多面体集合，表示循环 i in [0, N), j in [0, M)
    domain = isl.Set(f"[N, M] -> {{ S[i,j] : 0 <= i < N and 0 <= j < M }}")
    domain = domain.set_dim_id(isl.dim_type.param, 0, isl.Id('N')).set_dim_id(isl.dim_type.param, 1, isl.Id('M')).fix_val(isl.dim_type.param, 0, N).fix_val(isl.dim_type.param, 1, M)

    print("原始迭代空间（Polyhedral 表达）：")
    print(domain)

    # 应用一个简单的调度变换：交换 i 和 j 的顺序
    schedule = isl.Map("[N, M] -> { S[i,j] -> T[j,i] }")
    transformed = domain.apply(schedule)

    print("\n变换后的迭代空间（交换 i 和 j）：")
    print(transformed)

if __name__ == "__main__":
    main()
