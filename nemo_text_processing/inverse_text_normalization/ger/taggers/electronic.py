# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path, get_various_formats
from nemo_text_processing.text_normalization.de.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MIN_POS_WEIGHT,
    NEMO_ALPHA,
    GraphFst,
    capitalized_input_graph,
    insert_space,
)
from nemo_text_processing.text_normalization.de.utils import load_labels


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc. -> written form
    Handles letters, numbers, dots, slashes, dashes, underscores, umlaut normalization, multi-dot domains, www, http(s) and similar.
        e.g., b o b 28 at berlin punkt gov punkt d e -> tokens { electronic { username: "bob28" domain: "berlin.gov.de" } }
    Args:
        input_case: accepting both "cased" and "lower_cased" input.
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="electronic", kind="classify")

        # delete_extra_space = pynutil.delete(" ")

        SPACE = pynini.accept(" ")
        if input_case not in [INPUT_LOWER_CASED, INPUT_CASED]:
            raise ValueError(f'Unsupported input_case: {input_case}')
        
        SPACES = pynini.closure(SPACE)
        delete_extra_space = pynini.closure(SPACE, 0, 1)    

        OPT_SPACE = pynini.closure(SPACE, 0, 1)
        delete_extra_space = OPT_SPACE

        # letters
        letters = pynini.string_file(get_abs_path("data/electronic/letters.tsv")).invert()
        if input_case == INPUT_CASED:
            letters |= pynini.string_file(get_abs_path("data/electronic/capitalized_letters.tsv")).invert()
            letters = letters.optimize()

        # lower case letters
        lower_case_letters = pynini.string_file(get_abs_path("data/electronic/letters.tsv")).invert()
        lower_case_letters = lower_case_letters.optimize()
        NEMO_SPACE = pynini.accep(" ")

        # numbers
        num = pynini.string_file(get_abs_path("data/electronic/numbers.tsv")).invert()

        # Umlauts & esszet
        umlauts = pynini.string_file(get_abs_path("data/electronic/umlauts_esszet.tsv")).invert()

        # symbols
        symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).invert()
        if input_case == INPUT_CASED:
            symbols |= pynini.string_file(get_abs_path("data/electronic/capitalized_symbols.tsv")).invert()
            symbols = symbols.optimize()
        all_chars = letters | num | symbols | umlauts
        all_chars = all_chars.optimize()


        # if input_case == INPUT_CASED:
        #     num = capitalized_input_graph(num)

        # alpha_num = (NEMO_ALPHA | num).optimize()

        # url_symbols = pynini.string_file(get_abs_path("data/electronic/url_symbols.tsv")).invert()
        # accepted_username = alpha_num | url_symbols
        # process_dot = pynini.cross("dot", ".")
        # alternative_dot = (
        #     pynini.closure(delete_extra_space, 0, 1) + pynini.accep(".") + pynini.closure(delete_extra_space, 0, 1)
        # )
        # username = (alpha_num + pynini.closure(delete_extra_space + accepted_username)) | pynutil.add_weight(
        #     pynini.closure(NEMO_ALPHA, 1), weight=0.0001
        # )
        # username = pynutil.insert("username: \"") + username + pynutil.insert("\"")
        # single_alphanum = pynini.closure(alpha_num + delete_extra_space) + alpha_num
        # server = (
        #     single_alphanum
        #     | pynini.string_file(get_abs_path("data/electronic/server_name.tsv"))
        #     | pynini.closure(NEMO_ALPHA, 2)
        # )

        # if input_case == INPUT_CASED:
        #     domain = []
        #     # get domain formats
        #     for d in load_labels(get_abs_path("data/electronic/domain.tsv")):
        #         domain.extend(get_various_formats(d[0]))
        #     domain = pynini.string_map(domain).optimize()
        # else:
        #     domain = pynini.string_file(get_abs_path("data/electronic/domain.tsv"))

        # domain = single_alphanum | domain | pynini.closure(NEMO_ALPHA, 2)
        # domain_graph = (
        #     pynutil.insert("domain: \"")
        #     + server
        #     + ((delete_extra_space + process_dot + delete_extra_space) | alternative_dot)
        #     + domain
        #     + pynutil.insert("\"")
        # )
        # graph = username + delete_extra_space + pynutil.delete("at") + insert_space + delete_extra_space + domain_graph

        # ############# url ###
        # if input_case == INPUT_CASED:
        #     protocol_end = pynini.cross(pynini.union(*get_various_formats("www")), "www")

        #     protocol_start = pynini.cross(pynini.union(*get_various_formats("http")), "http") | pynini.cross(
        #         pynini.union(*get_various_formats("https")), "https"
        #     )
        # else:
        #     protocol_end = pynini.cross(pynini.union("w w w", "www"), "www")
        #     protocol_start = pynini.cross("h t t p", "http") | pynini.cross("h t t p s", "https")

        # protocol_start += pynini.cross(" colon slash slash ", "://")

        # # .com,
        # ending = (
        #     delete_extra_space
        #     + url_symbols
        #     + delete_extra_space
        #     + (
        #         domain
        #         | pynini.closure(
        #             accepted_username + delete_extra_space,
        #         )
        #         + accepted_username
        #     )
        # )

        # protocol_default = (
        #     (
        #         (pynini.closure(delete_extra_space + accepted_username, 1) | server)
        #         | pynutil.add_weight(pynini.closure(NEMO_ALPHA, 1), weight=0.0001)
        #     )
        #     + pynini.closure(ending, 1)
        # ).optimize()
        # protocol = (
        #     pynini.closure(protocol_start, 0, 1) + protocol_end + delete_extra_space + process_dot + protocol_default
        # ).optimize()

        # if input_case == INPUT_CASED:
        #     protocol |= (
        #         pynini.closure(protocol_start, 0, 1) + protocol_end + alternative_dot + protocol_default
        #     ).optimize()

        # protocol |= pynini.closure(protocol_end + delete_extra_space + process_dot, 0, 1) + protocol_default

        # protocol = pynutil.insert("protocol: \"") + protocol.optimize() + pynutil.insert("\"")
        # graph |= protocol

        # if input_case == INPUT_CASED:
        #     graph = capitalized_input_graph(graph, capitalized_graph_weight=MIN_POS_WEIGHT)

        # final_graph = self.add_tokens(graph)
        # self.fst = final_graph.optimize()


    def __init__(self, tn_electronic_tagger: GraphFst, tn_electronic_verbalizer: GraphFst, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        tagger = pynini.invert(tn_electronic_verbalizer.graph).optimize()
        verbalizer = pynini.invert(tn_electronic_tagger.graph).optimize()
        final_graph = tagger @ verbalizer

        graph = pynutil.insert("name: \"") + final_graph + pynutil.insert("\"")
        self.fst = graph.optimize()