《Automated Framework for Semantic Segmentation and CFD-Ready Geometric Reconstruction of Urban Elements from UAV Data》
Sustainable Cities and Society, 2025 (If 12.0)

# Abstract
Integrating semantic segmentation and geometric reconstruction into urban modeling is essential for CFD-based ventilation assessment in sustainable city design. Current pipelines are fragmented and force a trade-off between model detail and efficiency. To address this gap, we propose a dual-core automated framework based on UAV photogrammetry. Urban Semantic Segmentation (USS), a deep neural network with multi-dimensional feature extraction, segments 3D textured meshes of buildings, terrain, and vegetation. USS reaches the current state of the art on the SUM Helsinki dataset, achieving 93.74 % overall accuracy and 80.78 % mean class accuracy. After training on the SUM dataset, we applied the model to Melbourne’s CBD to validate its applicability across diverse urban morphologies. The Element Geometry Reconstruction (EGR) converted the segmentation outputs into watertight surface meshes. These meshes can directly generate high-quality computational grids. The reconstructed geometries show high agreement with real scenes: in the Melbourne CBD case, the median Hausdorff distance is 0.66 m for buildings and 0.43 m for terrain, and the median 2D canopy-projection offset is 2.51 m. The framework streamlines urban modeling, reducing preparation from days to hours, and provides a practical tool for high-fidelity CFD inputs to support evidence-based improvements in urban microclimates. However, challenges persist in assigning species-specific leaf area density (LAD) to every vegetation element at medium–large scales. Future work will tackle this limitation and explore aerial–ground image fusion–based modeling at small–to–medium scales to enhance the automated reconstruction of building-façade detail and near-ground element characteristics.
![论文总体框架20251022](https://github.com/user-attachments/assets/cf0de67e-364d-497a-b064-4956e175a1d0)

# Highlights
• USS network achieves 93.74 % overall accuracy on SUM Helsinki dataset.
• EGR engine generates CFD-ready multi-element urban meshes efficiently.
• Framework converts UAV photogrammetry to watertight 3D geometries.
• The framework balances detail and cost in vegetation reconstruction for CFD applications.
• The future study will assign species-specific leaf area density at medium–large scales.

# Upload
2025.11.19 Feature Extraction Code

# Citation
Please use the following citation when using the code.
Geng X, Gou Z. Automated Framework for Semantic Segmentation and CFD-Ready Geometric Reconstruction of Urban Elements from UAV Data[J]. Sustainable Cities and Society, 2025: 106948.
