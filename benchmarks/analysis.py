import pandas as pd

def analyze_benchmark(file_path, model_name):
    df = pd.read_csv(file_path, comment='#', 
                    names=['dtype','timestamp','ttft_ms','tokens','tps','peak_mem_mb','avg_mem_mb','prompt_chars','response_chars'])
    
    print(f"\n{model_name} Performance Summary:")
    print("-" * 40)
    
    # Iterate through records
    for index, row in df.iterrows():
        print(f"\nRecord {index+1}:")
        print(f"  Model: {row['dtype']}")
        print(f"  Timestamp: {row['timestamp']}")
        print(f"  TTFT: {row['ttft_ms']}ms")
        print(f"  Tokens: {row['tokens']}")
        print(f"  TPS: {row['tps']}")
        print(f"  Peak Memory: {row['peak_mem_mb']}MB")
        print(f"  Average Memory: {row['avg_mem_mb']}MB")
        print(f"  Prompt Length: {row['prompt_chars']}")
        print(f"  Response Length: {row['response_chars']}")

analyze_benchmark('benchmarks/llm_benchmark.log', 'Combined Models')
