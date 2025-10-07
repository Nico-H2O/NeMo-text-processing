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
import pdb 
# pdb.set_trace()
import pynini
from pynini import union, cross, closure
from pynini.lib import pynutil, rewrite

from nemo_text_processing.inverse_text_normalization.ger.graph_utils import( 
    GraphFst, NEMO_SPACE, INPUT_CASED, INPUT_LOWER_CASED, delete_space, delete_extra_space, insert_space,   
    NEMO_ALPHA, MIN_POS_WEIGHT, UMLAUTS,UMLAUT_MAP,UMLAUT_FST,NEMO_ALPHA_DE,
    capitalized_input_graph, 
    load_labels)

from nemo_text_processing.inverse_text_normalization.ger.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.en.utils import get_various_formats

# from nemo_text_processing.inverse_text_normalization.ger.taggers.cardinal import CardinalFst

class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. b o   b D R E I at g mail Sieben punkt ch -> tokens { electronic { username: "bob3" domain: "gmail7.ch" } }
    H T T P S doppelpunkt schrägstrich schrägstrich w w w punkt info sieben slash ü punkt Baden wuerttemberg
        -> tokens {electronic { protocol: "HTTPS://www.info7/ue.baden-wuerttemberg" } }
    Args:
        input_case: 
            accepting various input cases ('info', 'INFO', 'Info', 'i n f o', 'I N F O', 'I n f o').
            multi-word domain, 
            multiple spaces between words
            capital initial for: 
                domain (Info, Baden wuerttemberg), digit variants (Sieben, SIEBEN), W w w
            normalizes umlauts (option for server and domain names only commented out)   
            Capitalized token/syntagm not implemented for: Punkt, Doppelpunkt schrägstrich schrägstrich
    """
    # FST for German email normalization: handles letters, digits, "at" -> "@", "punkt" -> "."

    def __init__(self, input_case: str = INPUT_CASED):
        super().__init__(name="electronic", kind="classify")

        # ----------------------------------------
        ## 1) space handling
        # delete_extra_space = pynutil.delete(" ")
        delete_extra_space = pynini.closure(pynutil.delete(" "))
        
        # ----------------------------------------
        # 2) Letters: Consolidated German alphabet
        # to match one or more consecutive German letters (including umlauts)
        NEMO_ALPHA_DE = (NEMO_ALPHA | UMLAUTS).optimize()
        NEMO_ALPHA_DE_PLUS = NEMO_ALPHA_DE.plus.optimize() 

        # Extend NEMO_ALPHA to include the ASCII form of the German umlauts   
        NEMO_ALPHA_DE_FST = (NEMO_ALPHA | UMLAUT_FST).optimize()
        # NEMO_ALPHA_DE_FST_PLUS = NEMO_ALPHA_DE_FST.plus.optimize()  

        if input_case == INPUT_CASED:
            NEMO_ALPHA_DE = capitalized_input_graph(NEMO_ALPHA_DE)

        # -----------
        ## 3) numbers
        # cardinal_fst = CardinalFst().fst

        num = pynini.string_file(get_abs_path("data/cardinal/digits.tsv")) | pynini.string_file(
            get_abs_path("data/cardinal/zero.tsv")
        )

        if input_case == INPUT_CASED:
            num = capitalized_input_graph(num)
        
        alpha_num = (NEMO_ALPHA_DE_FST | num).optimize()

        digit_labels = []
        digits_list = load_labels(get_abs_path("data/cardinal/digits.tsv")) + load_labels(get_abs_path("data/cardinal/zero.tsv"))
        for d in digits_list:
            for variant in get_various_formats(d[0]):
                digit_labels.append((variant, d[1]))
                #digit_labels.append((" " + variant + " ", d[1]))
        spoken_digit_fst = pynini.string_map(digit_labels).optimize()

        # processing of "punkt" -> "."
        process_dot_spoken = (pynini.cross(" punkt ", ".") | pynini.cross(" dot ", ".")).optimize()
        alternative_dot = (
            pynini.closure(delete_extra_space, 0, 1) 
            + pynini.accep(".") 
            + pynini.closure(delete_extra_space, 0, 1)
        )
    
        # # ----------------------
        # # Email 1/3 - Local part

        # # loading URL symbols - accepted symbols for local part
        url_symbols = pynini.string_file(get_abs_path("data/electronic/url_symbols.tsv")).invert().optimize()
                
        # username_segment = pynutil.add_weight(spoken_digit_fst, 0.1) | alpha_num | url_symbols
        username_segment = alpha_num | url_symbols | spoken_digit_fst #| cardinal_fst # remove spoken_digit_fst?
        # username_segment = pynini.union(alpha_num, url_symbols, spoken_digit_fst, cardinal_fst)
        mixed_fst = pynini.closure(username_segment + delete_extra_space, 1)

        # semiotic tag wrapper for usernames
        username_tag = pynutil.insert('username: "') + mixed_fst + pynutil.insert('"')

        # # # # --------------------------------------------------------------------------------
        # # # #  Email 2/3 - Server part (eins und eins, n vidia, Y A N D E X,: l o l, a SIEBEN)
        # # # # --------------------------------------------------------------------------------

    # 1. Handling server_fst from TSV with all spoken variants
        server_labels = []
        for s in load_labels(get_abs_path("data/electronic/server_names.tsv")):
            if len(s) == 1:
                s = [s[0], s[0]]
            for variant in get_various_formats(s[0]):
                server_labels.append((variant, s[1]))
        server_fst = pynini.string_map(server_labels)

    # 2. Fallbacks for generic server names allowing sequences of German letters (with umlauts normalized) for cases not covered by the TSV mappings.
        generic_server = pynini.closure(NEMO_ALPHA_DE_FST, 1)
        single_alphanum = pynini.closure((alpha_num | spoken_digit_fst) + delete_extra_space) + (alpha_num | spoken_digit_fst)
        # single_alphanum = pynini.closure(alpha_num + delete_extra_space) + alpha_num

    # # 3. Server graph
        server_graph = (
            server_fst
        | pynutil.add_weight(generic_server, 10.0)
        | pynutil.add_weight(single_alphanum, 15.0)
        )

        # for no generic fallback paths
        # server_graph = server_fst #

        # # # ---------------------------------------------------------------------------------
        # # #  Email 3/3 - Domain part (O N L I N E, l o l, a SIEBEN, köln, Baden wuerttemberg)
        # # # ---------------------------------------------------------------------------------

    # 1. Build domain_fst from TSV with all spoken variants

        domain_labels = []
        for d in load_labels(get_abs_path("data/electronic/domains.tsv")):
            if len(d) == 1:
                d = [d[0], d[0]]
            for variant in get_various_formats(d[0]):
                domain_labels.append((variant, d[1]))
        domain_fst = pynini.string_map(domain_labels)

    # 2. Fallbacks for generic domains / could consolidate with server graph as generic_serv_dom
        generic_domain = pynini.closure(NEMO_ALPHA_DE_FST, 1)

    # 3. Domain graph
        domain_graph = (
            domain_fst
        | pynutil.add_weight(generic_domain, 10.0)
        | pynutil.add_weight(single_alphanum, 15.0)
        )

    ###### Combine server_graph and domain_graph with a dot in between
    # Complete domain graph (server + domain): eins und eins punkt a SIEBEN, Y A N D E X punkt köln, l o l punkt l o l)  

        # domain_tag = (
        #     pynutil.insert('domain: "')
        #     + server_graph               
        #     + delete_extra_space
        #     + ((delete_extra_space + process_dot_spoken + delete_extra_space) | alternative_dot)
        #     + delete_extra_space
        #     + domain_graph
        #     + pynutil.insert('"')
        # )

        domain_tag = (
            pynutil.insert('domain: "')
            + server_graph               
            + ((delete_extra_space + process_dot_spoken + delete_extra_space) | alternative_dot)
            + domain_graph
            + pynutil.insert('"')
        )

    # ###### Combine username, "at", and domain_tag (B o B null null SIEBEN at l o l punkt Baden wuerttemberg)
        # at_fst = pynini.delete(" at ", "@")
        
        email_graph = (
            username_tag
            + delete_extra_space
            + pynutil.delete("at")
            + insert_space
            + delete_extra_space
            + domain_tag
        )

# # multiple space, variant cases, umlauts normalization, multi-name domains
# # [INPUT]: b ö   SIEBEN   null at g mail punkt Baden wuerttemberg
# # [OUTPUT]: username: "boe70" domain: "gmail.baden-wuerttemberg"

# # # ########### url ###

        # loading URL symbols
        url_symbols = pynini.string_file(get_abs_path("data/electronic/url_symbols.tsv")).invert().optimize()

        http_protocol = (
            pynini.cross(pynini.union(*get_various_formats("http")), "http")
            | pynini.cross(pynini.union(*get_various_formats("https")), "https")
        )
        http_protocol += pynini.cross(
                pynini.union(" doppelpunkt doppelslash ", " doppelpunkt schrägstrich schrägstrich "), "://"
            )

###################### mapping "doppelpunkt + doppelslash / schrägstrich schrägstrich" -> "://".  ################
                     # unit and compund mapping removed from tsv (doppelpunkt doppelslash)

        # attempt at mapping "doppelpunkt schrägstrich schrägstrich" -> "://"
        # process_colon = pynini.cross(pynini.union(*get_various_formats("doppelpunkt")), ":")
        # process_slash = pynini.cross(pynini.union(*get_various_formats("schrägstrich")), "/")
        # process_double_slash = pynini.cross(pynini.union(*get_various_formats("doppelslash")), "//")
        # process_colon_slash_slash = (
        #     process_colon
        #     + delete_extra_space
        #     + (process_double_slash | (process_slash + delete_extra_space + process_slash))
        # )
 ############################################################################################################       

        # www_prefix. (possible upgrade: extend to other prefixes (vvw, ww2, etc.)
        www_prefix = pynini.cross(pynini.union(*get_various_formats("www")), "www")

        # public suffix: com, com.au, github.io, sa.edu.au, schools.nsw.edu.au
        ending_domain = domain_tag
        ending_username = pynini.closure(username_segment + delete_extra_space) + username_segment
        ending_core = ending_domain | ending_username

        ending = (
            delete_extra_space
                | pynini.closure(username_segment + delete_extra_space)
            + delete_extra_space
            + ending_core
        )

        # # test (at n vidia punkt de, slash a l i c i a null sieben, slash b o b D R E I, punkt l o l punkt Z ü r i c h (*Z not lowercased))

    # 1  # url startng without www_prefix - hostname only (Ö plus SIEBEN slash info punkt Baden wuerttemberg, a SIEBEN punkt eins und eins, Y A N D E X punkt köln, l o l punkt l o l)
         # e.g. bob.gmail.com, info.t-online.de => add a www_prefix, or that's on the client side?
        url_default = (
            (
                (pynini.closure(delete_extra_space + username_segment, 1) | server_graph)
                + delete_extra_space
                + ((delete_extra_space + process_dot_spoken + delete_extra_space) | alternative_dot)
                + delete_extra_space
                + domain_graph
            )
        )

        # slash_fst = pynini.closure(delete_extra_space + pynini.cross("schrägstrich", "/") + delete_extra_space, 0, 1)

        slash_fst = pynini.closure(delete_extra_space + pynini.cross("schrägstrich", "/") + delete_extra_space, 0, 1)
        path_segment = slash_fst + pynini.closure(alpha_num | url_symbols)
        path_fst = pynini.closure(path_segment)

        url_default = (
        # Optional slash before the domain/server
            pynini.closure(slash_fst, 0, 1)
            +
            (
                (pynini.closure(delete_extra_space + username_segment, 1) | server_graph)
                + delete_extra_space
                + ((delete_extra_space + process_dot_spoken + delete_extra_space) | alternative_dot)
                + delete_extra_space
                + domain_graph
                # Accept zero or more slashes and path segments after the domain
                + pynini.closure(slash_fst + NEMO_ALPHA_DE_FST.plus, 0, 1)
            )
        )

    # 2 # for the www prefix and hostname (www punkt l o l punkt de, www punkt Baden wuerttemberg)
        www_url = (
            pynini.closure(www_prefix + delete_extra_space + process_dot_spoken, 0, 1) + url_default
        )

    # 3 # for the complete generic url with protocol, prefix and hostname
        # ( H T T P S doppelpunkt schrägstrich schrägstrich W w w punkt Ö plus SIEBEN slash info punkt Baden wuerttemberg)
        url_main = (
            pynini.closure(http_protocol, 0, 1) 
            + delete_extra_space
            + pynini.closure(www_prefix, 0, 1)
            # + delete_extra_space 
            # + process_colon_slash_slash
            + delete_extra_space 
            + url_default
        )

        if input_case == INPUT_CASED:
            url_alt = (
                pynini.closure(http_protocol, 0, 1) 
                + delete_extra_space
                + www_prefix
                + delete_extra_space
                + alternative_dot 
                + delete_extra_space
                + url_default
            )
        else:
            url_alt = pynini.Fst()

        path_fst = pynini.accep("/") + pynini.closure(username_segment)

        url = (url_default | url_main | www_url)
        url_with_path = url + pynini.closure(path_fst, 0, 1)
        
        url_graph = (
            pynutil.insert('url: "') 
            + url_with_path
            + pynutil.insert('"')
        )

        self.fst = self.add_tokens(
            pynutil.add_weight(email_graph, 0.0) | pynutil.add_weight(url_graph, 1.0)
            ).optimize()

# # multi-name doamin normalizing.   
# # [INPUT]: H T T P S doppelpunkt schrägstrich schrägstrich W w w punkt Ö plus SIEBEN slash punkt Baden wuerttemberg
# # [OUTPUT]: electronic { protocol: "https://Www.oe+7/.baden-wuerttemberg" }

# FAILS
#1. Action: restrict substring match for: 
# example = "patricia at gmail punkt com"
# example = "heinz neun at gmail punkt com"
#2. Action: extend path segement for:
# example = "a b c punkt com slash d" # Pass: "a b c punkt com slash"
# example = "w w w punkt a b c punkt com slash d" # Pass: "w w w a b c punkt com slash"
example = "H T T P S doppelpunkt doppelslash W w w punkt a b c punkt com slash d" # Pass: "H T T P S doppelpunkt doppelslash W w w punkt a b c punkt com slash"
print(rewrite.top_rewrite(example, ElectronicFst().fst))


# ######################## TESTING ########################
# # PASS
# # example = "b o  b SIEBEN at gmail punkt com"
# # example = "b o  b SIEBEN at g mail punkt com"
# # example = "b o  b Sieben at gmail punkt com"
# # [OUTPUT]: username: "bob" domain: "gmail.com"
# # example = "b o  b slash Sieben at gmail punkt com"
# # example = "b o  b slash Sieben at g minus mail punkt com" -> url
# # example = "eins minus zwei minus drei punkt t v"
# # example = "patricia punkt moritz at t minus online punkt d e"
# example = "w w w punkt nordsee punkt com schrägstrich humus"
# # example = "impressum at congstar punkt de"
# # example = "H T T P S doppelpunkt schrägstrich schrägstrich W w w punkt Ö plus SIEBEN slash punkt Baden wuerttemberg"
# # example = "H T T P S doppelpunkt doppelslash W w w punkt Ö plus SIEBEN slash punkt Baden wuerttemberg"

# # [OUTPUT]{ url: "https://www.oe+7/.baden-wuerttemberg" }

# print(rewrite.top_rewrite(example, ElectronicFst().fst))

# # FAIL: capitals in url domain
# # example = "b o  b Sieben at gmail PUNKT com"
# # [OUTPUT]: username: "bob" domain: "gmail.com"