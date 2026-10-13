"""Streamlit interface: trained model prediction, structure, group annotations."""
import io
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from bace_core import predict_smiles

PROJECT_DIR = Path(__file__).resolve().parent


GROUP_SMARTS = {
    "苯环": "c1ccccc1", "羰基": "[CX3]=[OX1]", "酰胺": "[CX3](=[OX1])[NX3]",
    "羟基": "[OX2H]", "胺基（排除酰胺）": "[NX3;!$(N-C=O)]",
    "氟取代": "[#6]-[F]", "吡啶环": "n1ccccc1", "羧酸": "[CX3](=O)[OX2H1]",
}


@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


st.set_page_config(page_title="BACE-1 抑制剂预测", layout="wide")
st.title("BACE-1 抑制剂活性预测")
model_path = Path(st.sidebar.text_input("模型文件", str(PROJECT_DIR / "artifacts" / "model.joblib"))).expanduser()
if not model_path.is_file():
    st.info("请先完成模型训练，再选择生成的 model.joblib 文件。")
    st.stop()
try:
    artifact = load_model(str(model_path.resolve()), model_path.stat().st_mtime_ns)
except Exception as exc:
    st.error(f"模型无法加载：{exc}")
    st.stop()
model_identity = (str(model_path.resolve()), model_path.stat().st_mtime_ns)
if st.session_state.get("model_identity") != model_identity:
    st.session_state.pop("single_result", None)
    st.session_state.pop("batch_result", None)
    st.session_state["model_identity"] = model_identity
threshold = artifact["metadata"]["activity_threshold"]
st.caption(f"活性阈值 pIC50 ≥ {threshold:g}。预测用于候选分子筛选，实际活性需实验确认。")
single, batch = st.tabs(["单分子", "批量预测"])

with single:
    with st.form("single_prediction"):
        text = st.text_area("SMILES", placeholder="输入待预测分子的 SMILES")
        submitted = st.form_submit_button("预测活性")
    if submitted:
        try:
            st.session_state["single_result"] = predict_smiles(artifact, [text])
        except Exception as exc:
            st.session_state.pop("single_result", None)
            st.error(f"预测失败：{exc}")
    if "single_result" in st.session_state:
        try:
            row = st.session_state["single_result"].iloc[0]
            if row["error"]:
                st.error(row["error"])
            else:
                a, b, c = st.columns(3)
                a.metric("预测 pIC50", f"{row['pIC50_pred']:.3f}")
                b.metric("换算 IC50", f"{row['IC50_nM_pred']:.1f} nM")
                c.metric("活性判定", "活性" if row["active_pred"] else "非活性")
                mol = Chem.MolFromSmiles(row["canonical_SMILES"])
                matches, highlight = {}, set()
                for name, smarts in GROUP_SMARTS.items():
                    found = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
                    if found:
                        matches[name] = len(found)
                        highlight.update(atom for match in found for atom in match)
                st.image(Draw.MolToImage(mol, size=(700, 380), highlightAtoms=sorted(highlight)),
                         caption="二维结构；高亮部分匹配下方结构模式")
                st.write("结构模式匹配：", matches or "未匹配到预设模式")
                st.caption("模式可能重叠；这些标注不代表模型认定的因果药效基团。")
                with st.expander("三维构象", expanded=False):
                    try:
                        import py3Dmol
                        mol3d = Chem.AddHs(mol)
                        embed_options = AllChem.ETKDGv3()
                        embed_options.randomSeed = 42
                        status = AllChem.EmbedMolecule(mol3d, embed_options)
                        if status != 0:
                            st.info("当前分子未能生成三维构象。")
                        else:
                            if AllChem.MMFFHasAllMoleculeParams(mol3d):
                                AllChem.MMFFOptimizeMolecule(mol3d)
                            viewer = py3Dmol.view(width=650, height=420)
                            viewer.addModel(Chem.MolToMolBlock(mol3d), "mol")
                            viewer.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
                            viewer.zoomTo()
                            components.html(viewer._make_html(), height=440, scrolling=True)
                            st.caption("计算生成的单个构象；不代表与 BACE-1 结合时的真实姿态。")
                    except ImportError:
                        st.info("未安装三维展示依赖；运行 setup_visuals_windows.bat 后可用。")
                    except Exception as exc:
                        st.info(f"三维展示暂不可用：{exc}")
                with st.expander("本次预测的特征贡献", expanded=False):
                    try:
                        import shap
                        pipeline = artifact["pipeline"]
                        matrix = pipeline[:-1].transform(np.asarray([row["canonical_SMILES"]], dtype=object))
                        names = pipeline.named_steps["features"].get_feature_names_out()
                        names = names[pipeline.named_steps["variance"].get_support()]
                        explanation = shap.TreeExplainer(pipeline.named_steps["rf"])(matrix)
                        explanation.feature_names = list(names)
                        shap.plots.waterfall(explanation[0], max_display=12, show=False)
                        figure = plt.gcf()
                        st.pyplot(figure)
                        plt.close(figure)
                        st.caption("正值推高预测 pIC50，负值降低预测 pIC50；贡献不等于实验机制。")
                    except ImportError:
                        st.info("未安装 SHAP 依赖；运行 setup_visuals_windows.bat 后可用。")
                    except Exception as exc:
                        plt.close("all")
                        st.info(f"解释暂不可用：{exc}")
        except Exception as exc:
            st.error(f"预测或展示失败：{exc}")

with batch:
    uploaded = st.file_uploader("上传包含 SMILES 列的 CSV", type=["csv"])
    if uploaded is not None:
        try:
            import hashlib
            uploaded_bytes = uploaded.getvalue()
            upload_identity = hashlib.sha256(uploaded_bytes).hexdigest()
            if st.session_state.get("upload_identity") != upload_identity:
                st.session_state.pop("batch_result", None)
                st.session_state["upload_identity"] = upload_identity
            frame = pd.read_csv(io.BytesIO(uploaded_bytes))
            if "SMILES" not in frame.columns:
                st.error("CSV 必须包含名为 SMILES 的列。")
            elif st.button("开始批量预测"):
                with st.spinner("正在计算"):
                    st.session_state["batch_result"] = predict_smiles(artifact, frame["SMILES"].tolist())
            if "batch_result" in st.session_state:
                result = st.session_state["batch_result"]
                st.dataframe(result, use_container_width=True)
                st.download_button("下载预测 CSV", result.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="bace_predictions.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"CSV 读取或预测失败：{exc}")
