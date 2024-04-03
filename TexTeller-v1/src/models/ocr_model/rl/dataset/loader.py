from PIL import Image
from pathlib import Path
import datasets
import typst
import io

with open("mitex.typ", "w") as f:
    f.write("")
compiler = typst.Compiler("mitex.typ")


def mitex(latex, ppi=144.0):
    template = f"""
    #import "@preview/mitex:0.2.2": *
    #set page(height: auto, width: auto, margin: 0em)
    #mitex(`
    {latex}
    `)
    """
    with open("mitex.typ", "w") as f:
        f.write(template)
    return compiler.compile(format="png", ppi=ppi)


DIR_URL = Path(
    "/home/orangex4/projects/trl/TexTeller-v1/src/models/ocr_model/rl/dataset"
)
# e.g. DIR_URL = Path('/home/OleehyO/TeXTeller/src/models/ocr_model/train/dataset')


class LatexFormulas(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = []

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {"image": datasets.Image(), "latex_formula": datasets.Value("string")}
            )
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        dir_path = DIR_URL
        assert dir_path.is_dir()

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "dir_path": dir_path,
                },
            )
        ]

    def _generate_examples(self, dir_path: Path):
        formulas_path = dir_path / "im2latex_formulas.lst"

        i = 0
        with formulas_path.open("r", encoding="utf-8") as f:
            for latex_formula in f:
                try:
                    img_bytes = mitex(latex_formula)
                    yield str(i), {
                        # Image from bytes
                        "image": Image.open(io.BytesIO(img_bytes)),
                        "latex_formula": latex_formula.strip(),
                    }
                except Exception as e:
                    continue
                finally:
                    i += 1
                    # if i == 100:
                    #     break
