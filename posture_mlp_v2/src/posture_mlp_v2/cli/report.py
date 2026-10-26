"""Summarize frozen development outcomes without modifying model selection."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from ..data import read_jsonl,write_json,write_jsonl,key
from ..model import decide
from .train import detailed_metrics
from ..calibration import feasible


def pct(value):
    return "N/A" if value is None else f"{100*value:.1f}%"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--baseline-cache",type=Path)
    args=parser.parse_args()
    run=args.run_dir.resolve()
    read=lambda name:json.loads((run/name).read_text(encoding="utf-8"))
    audit,comparison,cal=read("data_audit.json"),read("development_comparison.json"),read("final_calibration.json")
    benchmark,consistency=read("benchmark.json"),read("annotation_consistency.json")
    chosen=comparison["selected"]
    selected=comparison["candidates"][chosen]
    cv=list(read_jsonl(run/"crossfit_predictions.jsonl"))
    selected_rows=[r for r in cv if r["candidate"]==chosen]
    policies={f["fold"]:f["policy"] for f in selected["folds"]}
    risk_curve=[]
    for minimum in (.65,.8,.9,.95,.975,.99,.995,.999,1.000001):
        predictions=[decide(r["prediction"],{**policies[r["fold"]],
                       "threshold":max(minimum,policies[r["fold"]]["threshold"])}) for r in selected_rows]
        risk_curve.append({"additional_minimum_confidence":minimum,
                           "note":"Only tightens each frozen fold policy; failed-closed folds stay rejected.",
                           **detailed_metrics(selected_rows,predictions)})
    write_json(run/"development_risk_coverage.json",risk_curve)
    baseline=None
    if args.baseline_cache:
        index={key(r):r for r in selected_rows}
        predicted={}
        for frame in read_jsonl(args.baseline_cache):
            for person in frame["people"]:
                k=(frame["output_frame"],person["person_id"])
                if k in index:
                    predicted[k]={**person["posture"]}
        if set(predicted)!=set(index):
            raise ValueError("Baseline cache does not cover comparison examples")
        baseline=detailed_metrics(selected_rows,[predicted[key(r)] for r in selected_rows])
        write_json(run/"frozen_v1_diagnostic.json",{
            "warning":"Not controlled or independent: v1 saw some old labels, new hard samples were mined using v1.",
            "cache":str(args.baseline_cache.resolve()),"metrics":baseline})
    m=selected["support_gated"]["overall"]
    operational=selected["support_gated"]["operational_risk_including_uncertain"]
    closed=[f["fold"] for f in selected["folds"] if f["policy"]["threshold"]>1]
    assessment={
        "status":"experimental_not_independently_validated","selected":chosen,
        "development_all_targets_met":feasible(m),"independent_test_available":False,
        "development_fail_closed_folds":closed,"production_ready":False,
        "labels_changed":0,
        "warnings":[
            "Old test and actively mined hard annotations are now development data, not a new test.",
            "Model family chosen using development cross-fit results; numbers are selection-biased.",
            "Final calibration block influenced development model selection; its metrics are NOT a final test.",
            "No persistent identity tracking: same people/camera recur across blocks.",
            "Five human uncertain instances remain accepted in the selected development predictions.",
            "Selective zero known error does NOT mean 100% overall accuracy or reliable unknown rejection.",
            "One development fold failed closed; temporal stability is inadequate.",
            "Manual clear group has no accepted known examples; do not hide this behind aggregate precision.",
        ],
    }
    write_json(run/"assessment.json",assessment)
    lines=[
        "# Hard 样本优化 v2：保守校准与基于邻域证据的拒识",
        "",
        "**结论：主要进展来自拒识机制，不是加深网络。当前仍为实验版，没有新的独立测试成绩。**",
        "",
        "## 数据核验",
        "",
        f"- hard导出共{audit['hard_export_n']}条，包含原{audit['old_n']}条＋新增{audit['new_n']}条；不是再把两份文件相加。",
        f"- 总标签：{audit['label_counts']}；新增：{audit['new_label_counts']}。",
        f"- 原标签改写记录：{len(audit['annotation_changes'])}条。本次不自动纠正人工标签。",
        "- 新增95条人工遮挡标签：clear16、partial25、severe54；其中severe standing共5条。",
        "- 已逐项核验两版缓存及标注；完整5394帧的框、置信度、关键点不变，未重新运行YOLO。",
        "",
        "## 评估边界与分割",
        "",
        "这96条新样本由旧模型分歧/错误附近主动筛选，旧测试集也已经用于分析。"
        "**全部只算开发数据**。另一个原视频或从未参与挖掘/分析的新时段才能做最终测试。",
        "",
        "每个60秒片段分为A=[0,17)、B=[23,37)、C=[43,60]；中间共12秒为缓冲，"
        "同一帧所有人同组。三轮依次(A训练/B校准/C验证)、(B训练/C校准/A验证)、(C训练/A校准/B验证)。"
        "每个纳入开发验证的样本只在其模型未训练、未校准的一轮评估；模型家族最后用这些开发结果选择。",
        f"实际跨组最小时间间隔：{audit['actual_minimum_gaps_seconds']} 秒。",
        "",
        "| 分块 | sitting | standing | uncertain | ignore |",
        "|---|---:|---:|---:|---:|",
    ]
    for b,c in audit["block_label_counts"].items():
        lines.append(f"| {b}（-1为缓冲） | {c.get('sitting',0)} | {c.get('standing',0)} | {c.get('uncertain',0)} | {c.get('ignore',0)} |")
    lines += [
        "",
        "缓冲区有76条标注，不参与训练、阈值选择或分块评估。因数据同一视频、相同人员且主动抽样，"
        "以上既不是跨人物测试，也不是随机自然分布性能。第2400帧本轮在缓冲区，但仍是已知开发案例。",
        "",
        "## 模型与拒识改动",
        "",
        "- 保留251维遮挡感知特征：躯干中心/尺度归一化、置信度、缺失掩码、几何与可见度；不输入绝对坐标、框位置、人员编号或时间。",
        "- 同数据对比：原大小MLP(32/16)、更强正则小MLP(16/8、dropout0.35、标签平滑)、线性二分类器。",
        "- 校准比较T=1与T∈[1,8]，禁止T<1放大置信度；本轮选中T=1，不再锐化概率。",
        "- 仍只训练sitting/standing。uncertain不变成第三个softmax类别，而是训练集的拒识参考。",
        "- 新的拒识证据：与人工uncertain姿态相近、远离训练同类姿态、最近已知类别与模型矛盾、镜像两视图类别不一致、骨架不足。",
        "- 邻域参考只来自当前训练块；镜像不重复计为两个人；支持半径排除3秒内相邻样本。"
        "推理不使用时间/位置/人工标签查表，只比较归一化特征。",
        "- 在校准块扫描有限阈值网格，目标为已判定站坐经验错误率≤3%、人工uncertain拒识≥90%、"
        "至少接受10个明确样本且每类至少预测2个；没有可行点则全部unknown。",
        "- 参数/模型家族在实验前固定，首次执行仅因NumPy布尔值JSON序列化失败；修复序列化后按相同参数重跑，未按结果调参。",
        "",
        "## 三轮时间分块开发结果",
        "",
        "| 候选 | 已判定/明确样本 | unknown比例 | 已判定错误 | uncertain拒识 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name,value in comparison["candidates"].items():
        v=value["support_gated"]["overall"]
        lines.append(f"| {name}{'（暂选）' if name==chosen else ''} | {v['accepted']}/{v['known_n']} | "
                     f"{pct(v['unknown_rate'])} | {v['errors_among_accepted']} | "
                     f"{v['uncertain_rejected']}/{v['uncertain_n']} ({pct(v['uncertain_rejection_rate'])}) |")
    lines += [
        "",
        "**所有候选均未同时满足汇总开发结果的全部目标。** 暂选模型是按预先规定的"
        "“目标违约最小，再看覆盖率”顺序得到的最接近目标者，不代表已经达标。",
        "被拒识样本仍计入召回率分母；0个站坐误判不等于100%准确率。",
        f"把误接受uncertain也算为不安全接受时：{operational['unsafe_acceptances']}/{operational['accepted']}，"
        f"比例{pct(operational['rate'])}。uncertain无确定站坐真值，这个指标是保守的操作风险，不是事实错误率。",
        "",
        "| 暂选模型类别 | 真值数量 | 精确率 | 召回率（拒识算漏判） | unknown |",
        "|---|---:|---:|---:|---:|",
    ]
    for c,v in m["per_class"].items():
        lines.append(f"| {c} | {v['support']} | {pct(v['precision'])} | {pct(v['recall'])} | {v['unknown']} |")
    lines += [
        "",
        "### 分轮稳定性（不能只看汇总）",
        "",
        "| 轮次 | 校准阈值 | 接受/明确样本 | 错误 | uncertain拒识 |",
        "|---|---:|---:|---:|---:|",
    ]
    for f in selected["folds"]:
        v=f["evaluation"]["overall"]
        lines.append(f"| {f['fold']} | {f['policy']['threshold']:.6f} | {v['accepted']}/{v['known_n']} | "
                     f"{v['errors_among_accepted']} | {v['uncertain_rejected']}/{v['uncertain_n']} |")
    lines += [
        "",
        f"轮次{closed}无可行校准点而全部拒识；因此跨时段覆盖率仍不稳定，不能当作部署保证。",
        "",
        "### 只用概率阈值的对照",
        "",
        "| 模型 | 接受/明确样本 | 已判定错误 | uncertain拒识 |",
        "|---|---:|---:|---:|",
    ]
    for name,value in comparison["candidates"].items():
        v=value["probability_only"]["overall"]
        lines.append(f"| {name} | {v['accepted']}/{v['known_n']} | {v['errors_among_accepted']} | "
                     f"{v['uncertain_rejected']}/{v['uncertain_n']} |")
    lines += [
        "",
        "该对照使用相同校准目标；多组只能通过全部拒识保证条件，说明单纯把概率门槛拉高不足。"
        "不同网格/样本会影响结论，本轮不据开发答案继续增大搜索。",
        "",
        "## 人工遮挡分组（暂选模型）",
        "",
        "| 遮挡 | sitting/standing真值 | 接受/明确样本 | 站坐错误 | uncertain拒识 |",
        "|---|---:|---:|---:|---:|",
    ]
    for group,v in selected["support_gated"]["manual_occlusion_groups"].items():
        lines.append(f"| {group} | {v['per_class']['sitting']['support']}/{v['per_class']['standing']['support']} | "
                     f"{v['accepted']}/{v['known_n']} | {v['errors_among_accepted']} | "
                     f"{v['uncertain_rejected']}/{v['uncertain_n']} |")
    lines += [
        "",
        "**clear组6个明确站立样本全部被拒识**；严重遮挡站立进入开发验证的只有2个，接受1个、拒识1个。"
        "所以不能只说强遮挡样本无误判，也不能认为遮挡问题已经解决。完整组内precision/recall/F1和可见度代理分组见development_comparison.json。",
        "",
        "## 最终实验模型（不是新的独立评估）",
        "",
        f"模型家族={chosen}；重新以A+C训练、B校准；温度={cal['actual_temperature']:.3f}；"
        f"拒识策略={cal['policy_selection']['chosen']['policy']}。",
        "保持原大小MLP不是拒绝正则化，而是本数据上更小模型没有改善uncertain拒识与覆盖率的折中。",
        "",
    ]
    final=cal["policy_selection"]["chosen"]["metrics"]
    lines.append(f"最终校准块明确样本接受{final['accepted']}/{final['known_n']}、错误{final['errors_among_accepted']}；"
                 f"uncertain拒识{final['uncertain_rejected']}/{final['uncertain_n']}。"
                 "这是用于选择阈值的同一校准集，不能当作独立性能。没有再把B块纳入训练重拟合。")
    lines += [
        "",
        "已知帧1619/1709的P1恢复standing，2400的P1为unknown且P2/P3/P4为sitting；"
        "5125/5215/5305的P1为unknown。但除2400为已知开发缓冲帧外，其余属于最终训练块，"
        "这些只作回归展示，不能用于证明泛化。",
        f"CPU特征＋镜像＋分类＋邻域拒识约{benchmark['median_ms_per_person']:.2f} ms/人（批处理；不含YOLO/解码/画图）。"
        "部署仅需要NumPy，NPZ参考库不含视频图像。",
        "",
        "## 弯腰站立与uncertain标注复核",
        "",
        "标准仍为：明确站立（含弯腰）标standing；当前单帧看不清支撑关系/起坐过渡/半蹲歧义标uncertain。"
        "不因躯干倾斜直接标uncertain，也不因腿可见直接标standing。",
        "生成了10对疑似近邻冲突图，**没有改标签**。例：F330/P1标standing，F3866/P1、F5035/P1标uncertain；"
        "F4170/P2标standing，F4226/P2标uncertain。需要人结合支撑关系判断，骨架相近不证明任何一方错误。",
        "",
        "复核图：[第一页](label_review/consistency_page_1.jpg)、[第二页](label_review/consistency_page_2.jpg)；"
        "完整样本清单为annotation_consistency.json。",
        "",
        "## 下一步",
        "",
        "1. 先统一上述标注边界；复核优先，不自动把全部uncertain改为standing。",
        "2. 增加不参与旧模型挖掘的新视频/不相邻新时段作为最终测试，冻结本轮选择后再评估。"
        "补充不同人物和位置的遮挡站立、腿可见坐姿。",
        "3. 目前问题包括标签语义重叠、部分可见站立拒识、跨时段拒识过多，并非只集中在强遮挡；"
        "因此本轮不增加遮挡双分支或图像网络。若独立数据确认剩余错误主要来自骨架信息缺失，再比较图像辅助。",
        "",
        "参考：[LAViTSPose](https://pmc.ncbi.nlm.nih.gov/articles/PMC12732004/)遮挡感知思路；"
        "[SitPose](https://arxiv.org/abs/2412.12216)轻量分类思路。不是论文流水线复现，也不外推其3D骨架性能。",
        "",
    ]
    if baseline:
        old=baseline["by_origin"]["hard_new"]["overall"]
        new=selected["support_gated"]["by_origin"]["hard_new"]["overall"]
        lines += ["## 冻结v1的困难开发样本诊断（非严格对照）","",
                  "只看同一批纳入分块验证的新hard样本；旧模型从未训练这些标签，但它参与了样本挖掘，仍有选择偏差。",
                  "",
                  "| 模型 | 接受/明确样本 | 站坐错误 | uncertain拒识 |",
                  "|---|---:|---:|---:|",
                  f"| 冻结v1 | {old['accepted']}/{old['known_n']} | {old['errors_among_accepted']} | {old['uncertain_rejected']}/{old['uncertain_n']} |",
                  f"| v2逐块未见开发预测 | {new['accepted']}/{new['known_n']} | {new['errors_among_accepted']} | {new['uncertain_rejected']}/{new['uncertain_n']} |",""]
    (run/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"report":str(run/"REPORT.md"),"assessment":assessment},indent=2))


if __name__=="__main__":
    main()
