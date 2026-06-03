import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm

# Find available CJK fonts
fonts = sorted(set([f.name for f in fm.fontManager.ttflist]))
cjk = [f for f in fonts if any(k in f.lower() for k in ['yahei','hei','song','ming','fang','kai','noto','wenquan','source han','pingfang','simsun','simhei','deng','microsoft ya'])]
print("CJK fonts found:", cjk)
for n in ['SimHei', 'Microsoft YaHei', 'Microsoft YaHei UI', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DengXian', 'Noto Sans SC']:
    print(f"  {n}: {'FOUND' if any(n in f for f in fonts) else 'NOT FOUND'}")
print("\n--- First 80 fonts ---")
for f in fonts[:80]:
    print(f"  {f}")
