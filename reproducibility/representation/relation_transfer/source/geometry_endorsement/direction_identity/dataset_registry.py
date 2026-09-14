from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    aries_name: str
    domain: str
    citation: str
    official_source: str
    license_classification: str
    license_source: str
    predefined_train_dev_test: bool
    orientation: str | None
    support_count: int
    attack_count: int
    group_unit: str | None
    english: bool = True
    pair_text_sufficient: bool = True
    locally_usable: bool = True
    notes: str = ""

    @property
    def min_binary_class(self) -> int:
        return min(self.support_count, self.attack_count)

    def eligibility_failures(self) -> list[str]:
        failures: list[str] = []
        if not self.english: failures.append("NOT_ENGLISH")
        if not self.official_source: failures.append("NO_OFFICIAL_SOURCE")
        if not self.locally_usable: failures.append("LICENSE_NOT_LOCALLY_USABLE")
        if not self.orientation: failures.append("ORIENTATION_UNVERIFIED")
        if not self.group_unit: failures.append("GROUP_ID_UNAVAILABLE")
        if not self.pair_text_sufficient: failures.append("PAIR_CONTEXT_INSUFFICIENT")
        if self.support_count < 100: failures.append("SUPPORT_COUNT_BELOW_100")
        if self.attack_count < 100: failures.append("ATTACK_COUNT_BELOW_100")
        return failures

    def record(self) -> dict[str, Any]:
        value = asdict(self)
        value.update({"minimum_binary_class_count": self.min_binary_class,
                      "eligible": not self.eligibility_failures(),
                      "eligibility_failures": self.eligibility_failures()})
        return value


ARIES_CITATION = "Gemechu, Ruiz-Dolz & Reed (2024), doi:10.18653/v1/2024.argmining-1.1"


def official_aries_registry() -> tuple[DatasetSpec, ...]:
    orientation = "proposition_1=premise/source; proposition_2=claim/target; inference=support; conflict=attack"
    return (
        DatasetSpec("MTC", "Microtext", "structured_argumentation",
            "Peldszus & Stede (2015); Stede et al. (2016)",
            "https://github.com/peldszus/arg-microtexts-multilayer",
            "CC-BY-NC-SA-4.0", "official repository LICENSE/README", False,
            orientation, 272, 108, "microtext document"),
        DatasetSpec("AAEC", "AAEC", "persuasive_essay",
            "Stab & Gurevych (2017), doi:10.1162/COLI_a_00295",
            "https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2421",
            "OPEN_ACCESS_LICENSE_TEXT_UNRESOLVED", "TUdatalib item metadata", False,
            orientation, 4841, 497, "essay", locally_usable=False,
            notes="Open-access label is not a concrete reuse license; excluded fail-closed."),
        DatasetSpec("CDCP", "CDCP", "financial_consumer_comments",
            "Park & Cardie (2018)", "https://facultystaff.richmond.edu/~jpark/data/cdcp_acl17.zip",
            "SOURCE_TERMS_UNRESOLVED", "original corpus release", False,
            orientation, 694, 82, "comment document", locally_usable=False),
        DatasetSpec("ACSP", "ACSP", "scientific_papers",
            "Lauscher et al. (2018)", "https://github.com/UKPLab/coling2018-xling_argument_mining",
            "MIXED_SOURCE_TEXT_RIGHTS_UNRESOLVED", "original corpus release", False,
            orientation, 8069, 697, "scientific paper", locally_usable=False),
        DatasetSpec("AMP", "AMP", "online_discussion",
            "Chakrabarty et al. (2019), doi:10.18653/v1/D19-1291",
            "https://github.com/CogComp/ArgumentMining-AMPERSAND",
            "SOURCE_TERMS_UNRESOLVED", "original release", False,
            None, 2111, 0, "discussion thread", locally_usable=False,
            notes="ARIES encodes relation-present versus neutral, not SUPPORT versus ATTACK."),
        DatasetSpec("AbstRCT", "ABstRACT", "medical_abstracts",
            "Mayer, Cabrio & Villata (2020)", "https://gitlab.com/tomaye/abstrct",
            "CC-BY-NC-SA-4.0", "official repository README/LICENSE", True,
            "BRAT Support/Attack Arg1=source and Arg2=target", 2290, 344,
            "PubMed abstract/document ID",
            notes="Official BRAT Partial-Attack and Attack are both explicit negative/attack relations; both map to the frozen attack class."),
        DatasetSpec("US2016", "US2016", "political_debate",
            "Visser et al. (2020)", "https://corpora.aifdb.org/US2016",
            "ACADEMIC_RESEARCH_ONLY", "AIFdb corpus terms: free for academic use", False,
            orientation, 2765, 866, "debate argument map"),
        DatasetSpec("QT30", "Cuties", "broadcast_question_time",
            "Hautli-Janisz et al. (2022), ACL Anthology 2022.lrec-1.352",
            "https://corpora.aifdb.org/qt30", "ACADEMIC_RESEARCH_ONLY",
            "AIFdb corpus terms: free for academic use", False, orientation,
            2714, 545, "Question Time episode/argument map"),
    )


def rank_eligible(specs: tuple[DatasetSpec, ...] | None = None) -> list[DatasetSpec]:
    eligible = [s for s in (specs or official_aries_registry()) if not s.eligibility_failures()]
    return sorted(eligible, key=lambda s: (
        -int(s.predefined_train_dev_test), -int(bool(s.orientation)),
        -s.min_binary_class, s.dataset_id.lower()))


def select_secondaries(specs: tuple[DatasetSpec, ...] | None = None) -> tuple[DatasetSpec, DatasetSpec]:
    ranked = rank_eligible(specs)
    if not ranked:
        raise RuntimeError("INSUFFICIENT_EXTERNAL_DATASET_DIVERSITY")
    first = ranked[0]
    second = next((item for item in ranked[1:] if item.domain != first.domain), None)
    if second is None:
        raise RuntimeError("INSUFFICIENT_EXTERNAL_DATASET_DIVERSITY")
    return first, second
