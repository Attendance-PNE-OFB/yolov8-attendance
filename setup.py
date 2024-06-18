import setuptools

setuptools.setup(
    name="yolov8_attendance",
    version="2.0.0",
    description="Script de classification d'image basé sur le modèle yolov8",
    maintainer="Parc national des Écrins",
    url="https://github.com/PnEcrins/yolov8-attendance",
    packages=setuptools.find_packages(where="."),
    package_data={"model": ["model/yolov8x.pt"]},
    install_requires=(list(open("requirements.txt", "r"))),
    extras_require={
        "dev": [
            "black",
        ],
    },
    python_requires=">=3.9",
)
