## 2026-09-14T03:27:31Z

You are the Project Orchestrator for this task, operating via the SWE Light workflow.

Workspace Directory: E:/PythonFile/Project/Low-Light-Image-Enhancement
Your Working Directory: E:/PythonFile/Project/Low-Light-Image-Enhancement/.agents/teamwork_preview_swe_1
Original Request Path: E:/PythonFile/Project/Low-Light-Image-Enhancement/.agents/ORIGINAL_REQUEST.md

Please read E:/PythonFile/Project/Low-Light-Image-Enhancement/.agents/ORIGINAL_REQUEST.md for the complete specification.
Summary of task:
Implement an Illumination-Guided Delta Modulation Mamba module (Method 1) for the HV branch in the Low-Light Image Enhancement (HVI-CIDNet) project, where features from branch I act as an accelerator/brake controlling the step size Delta of Mamba scanning over branch HV.
- Standalone PyTorch module: `src/HVI-CIDNet/net/IG_Mamba.py`
- Seamless dimension and spatial preservation (4-direction scanning / SS2D)
- Explanation and unit test script verifying forward pass, gradient flow, and edge cases ($I \to 0$).

Maintain your BRIEFING.md and progress.md in your working directory. Regularly update progress.md. When the implementation and review loop is complete and verified with tests, send your final completion report back to me.
