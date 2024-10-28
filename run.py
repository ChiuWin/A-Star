# run_example.py

times = 50
for i in range(1, times):
    print(f"运行第 {i} 次")
    import os
    os.system(f'python demo3.py {i}')
