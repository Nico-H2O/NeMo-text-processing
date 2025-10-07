# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import pynini 
#from pynini import FAR
from pynini.lib import pynutil, rewrite

from nemo_text_processing.inverse_text_normalization.ger.taggers.cardinal import (
    CardinalFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.ordinal import (
    OrdinalFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.decimal import (
    DecimalFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.electronic import (
    ElectronicFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.fraction import (
    FractionFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.date import (
    DateFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.time import (
    TimeFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.money import (
    MoneyFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.measure import (
    MeasureFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.telephone import (
    TelephoneFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.punctuation import (
    PunctuationFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.word import WordFst
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)

class ClassifyFst(GraphFst):
    def __init__(
        self,
        cache_dir = None,
        whitelist = None,
        overwrite_cache: bool = False,
        input_case: str = INPUT_LOWER_CASED,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify")

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"ger_itn_{input_case}.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
        else:
            cardinal = CardinalFst()
            cardinal_graph = cardinal.fst

            ordinal = OrdinalFst(cardinal)
            ordinal_graph = ordinal.fst

            decimal = DecimalFst(cardinal)
            decimal_graph = decimal.fst

            electronic_graph = ElectronicFst().fst

            fraction = FractionFst(cardinal)
            fraction_graph = fraction.fst

            date = DateFst(cardinal, ordinal)
            date_graph = date.fst

            time = TimeFst(cardinal)
            time_graph = time.fst

            money = MoneyFst(cardinal)
            money_graph = money.fst

            measure = MeasureFst(cardinal, decimal, fraction)
            measure_graph = measure.fst

            telephone = TelephoneFst(cardinal)
            telephone_graph = telephone.fst

            word_graph = WordFst().fst
            punct_graph = PunctuationFst().fst

            classify = (
                pynutil.add_weight(cardinal_graph, 1.0)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(electronic_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.11)
                | pynutil.add_weight(time_graph, 1.12)
                | pynutil.add_weight(money_graph, 1.1)
                | pynutil.add_weight(measure_graph, 1.1)
                | pynutil.add_weight(telephone_graph, 1.1)
                | pynutil.add_weight(word_graph, 100)
            )

            punct = (
                pynutil.insert("tokens { ")
                + pynutil.add_weight(punct_graph, weight=1.1)
                + pynutil.insert(" }")
            )
            # self.fst = punct

            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            # self.fst = token

            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" "))
                + token
                + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(
                delete_extra_space + token_plus_punct
            )
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})



############ tests
# CASE 1: PASS
# example = "eins minus zwei minus drei punkt t v"
# print(rewrite.top_rewrite(example, ElectronicFst().fst))
# OUTPUTS: electronic { url: "1-2-3.tv" } # PASS with all graphs on

# CASE 1b PASS
# example = "eins minus zwei minus drei punkt t v"
# example = "eins minus punkt t v"
# print(rewrite.top_rewrite(example, ElectronicFst().fst))
# OUTPUTS: electronic { url: "1-.tv" }

# Case 1c: PASS as cardinal
# example = "eins minus"
# clf = ClassifyFst()
# print(rewrite.top_rewrite(example, clf.fst))
# OUTPUTS: tokens { cardinal { integer: "eins" } } tokens { name: "minus" }

# Q: Is this pasing as a single token or is it split 

# but failed ?? with:
# example = "eins minus zwei minus drei punkt t v"
# clf = ClassifyFst()
# print(rewrite.top_rewrite(example, clf.fst))
# outputs: tokens { electronic { url: "1-2-3.tv" } }
# -> interference from other graphs
#############
# CASE 2: all fields active
# example = "eins"
# clf = ClassifyFst()
# print(rewrite.top_rewrite(example, clf.fst))
# OUTPUTS:
# tokens { cardinal { integer: "eins" } }
# tokens { telephone { number_part: "1" } } # with cardinal off

###########
# CASE 3: FAIL for ElectronicFst. passes for ClassifyFst
# example = "eins minus" # composition failure
# example = "eins punkt" # composition failure

# CASE 3a FAIL to map eins to 1
# example = "eins"
# clf = ClassifyFst()
# print(rewrite.top_rewrite(example, clf.fst))
# tokens { cardinal { integer: "eins" } }

# tokens { electronic { url: "1" } } # with cardinal off & on ??

# CASE 3b
# example = "punkt" # => 
# clf = ClassifyFst()
# print(rewrite.top_rewrite(example, clf.fst))
# # tokens { name: "punkt" }

###########
# CASE 4: FAIL
# example = "eins punkt zwei" # composition failure
# clf = ClassifyFst()
# print(rewrite.top_rewrite(example, clf.fst))
# tokens { electronic { url: "1.zwei" } } # with cardinal off & on : matches too many other graphs (cardinal, word, punctuation)

# example = "eins minus zwei" # composition failure
# clf = ClassifyFst()
# print(rewrite.top_rewrite(example, clf.fst))
# #Outputs: tokens { cardinal { integer: "eins" } } tokens { cardinal { negative: "-" integer: "zwei" } }

# example = "eins komma zwei" # composition failure
# clf = ClassifyFst()
# print(rewrite.top_rewrite(example, clf.fst))
#Outputs: tokens { decimal { integer_part: "1" fractional_part: "2" } }

###########
# CASE 5: URL FAIL & EMAIL PASS 
# 2 outputs: URL FAILS to match correctly. URL are favored over emails?
# example = "eins punkt zwei at gmail punkt com" 
# clf = ClassifyFst()
# print(rewrite.top_rewrite(example, clf.fst))
# 2 outputs:
# tokens { electronic { url: "1.zwei" } } # URL FAILS to match correctly. URL are favored over emails?
# tokens { electronic { username: "1.2" domain: "gmail.com" } }
# only email match after changing weights:
# outputs: tokens { electronic { username: "1.2" domain: "gmail.com" } }


# CASE 6: EMAIL PASS
# example = "eins minus zwei at gmail punkt com" 
# clf = ClassifyFst()
# print(rewrite.top_rewrite(example, clf.fst))
# outputs: tokens { electronic { username: "1-2" domain: "gmail.com" } }

# CASE 7: URL PASS
# example = "eins minus zwei minus drei punkt tv at gmail punkt com" 
# clf = ClassifyFst()
# print(rewrite.top_rewrite(example, clf.fst))
#outputs: tokens { electronic { username: "1-2-3.tv" domain: "gmail.com" } }

# CASE 8: PASS
# example = "eins minus zwei punkt tv punkt com" 
# clf = ClassifyFst()
# print(rewrite.top_rewrite(example, clf.fst))
# outputs: tokens { electronic { url: "1-2.tv.com" } }

# CASE 9: PASS
# example = "eins minus zwei punkt t v" 
# clf = ClassifyFst()
# print(rewrite.top_rewrite(example, clf.fst))
# outputs: tokens { electronic { url: "1-2.tv" } }

# CASE 10: PASS
# example = "eins minus zwei punkt tv" 
# clf = ClassifyFst()
# print(rewrite.top_rewrite(example, clf.fst))
# outputs: tokens { electronic { url: "1-2.tv" } }

# CASE 11: PASS
# example = "eins minus zwei minus drei punkt tv" 
# clf = ClassifyFst()
# print(rewrite.top_rewrite(example, clf.fst))
# outputs: tokens { electronic { url: "1-2-3.tv" } }



#####
#CASE 12:
# example = "eins - t v"
# clf = ClassifyFst()
# graph = clf.fst
# print(rewrite.top_rewrite(example, clf.fst))

# all active graphs:
# OUTPUT: tokens { telephone { country_code: "1" number_part: "" } } tokens { name: "-" } tokens { name: "t" } tokens { name: "v" }

# same outputs with:
# example = "eins."
# clf = ClassifyFst()
# graph = clf.fst
# print(rewrite.top_rewrite(example, graph))  # Should output a tagged cardinal followed by tagged punctuation
# tokens { cardinal { integer: "eins" } } tokens { name: "." } without space between eins and .
# tokens { telephone { country_code: "1" number_part: "" } } tokens { name: "." }

# input_text = 'eins .' # with space inbetween
# clf = ClassifyFst()
# token_plus_punct = clf.fst
# print(rewrite.top_rewrite(input_text, token_plus_punct))
# OUTPUT: tokens { telephone { country_code: "1" number_part: "" } } tokens { name: "." }

###############
# SIDE NOTE: punctuation alone
# punct_graph = PunctuationFst().fst
# punct = (
#     pynutil.insert("tokens { ")
#     + pynutil.add_weight(punct_graph, weight=1.1)
#     + pynutil.insert(" }")
# )

# side note: in PunctuationFST
# clf = ClassifyFst()
# example = "-"
# print(rewrite.top_rewrite(example, clf.fst))
# outputs: tokens { name: "-" } but that's PunctuationFst
############

##########
# example = "eins . t v"            

# print(rewrite.top_rewrite(example, ElectronicFst().fst))  # Should output a tagged cardinal
            # print(rewrite.top_rewrite(".", punct))
            # print(rewrite.top_rewrite("eins .", graph))  # Should output a tagged cardinal followed by tagged punctuation
            # print(rewrite.top_rewrite("eins minus zwei minus drei punkt t v", graph))
            # print(rewrite.top_rewrite("eins minus zwei minus drei punkt t v .", graph))
           