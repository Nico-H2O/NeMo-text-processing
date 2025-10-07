from symtable import SymbolTable
import pynini, string
from pynini.lib import pynutil, rewrite
from pynini import SymbolTable

from nemo_text_processing.inverse_text_normalization.ger.graph_utils import( 
    GraphFst, delete_space,delete_extra_space, insert_space, NEMO_NOT_QUOTE, NEMO_ALPHA_DE)
from nemo_text_processing.inverse_text_normalization.ger.utils import get_abs_path

class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic fields.
    Currently supports emails (username + domain) and URLs (url field).
    e.g. tokens { electronic { username: "bob3" domain: "gmail.com" } } -> bob3@gmail.com
         tokens { electronic { url: "www.example.com" } } -> www.example.com
    """

    def __init__(self):
        super().__init__(name="electronic", kind="verbalize")

        user_name = (
            pynutil.delete("username:")
            + pynini.closure(delete_space, 0, 1)
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + pynini.closure(delete_space, 0, 1)
        )

        domain = (
            pynutil.delete("domain:")
            + pynini.closure(delete_space, 0, 1)
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + pynini.closure(delete_space, 0, 1)
        )

        email = (
            user_name
            + pynini.closure(delete_space, 0, 1)
            + pynutil.insert("@")
            + pynini.closure(delete_space, 0, 1)
            + domain
        )

        protocol = (
            pynutil.delete("url:")
            + pynini.closure(delete_space, 0, 1)
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + pynini.closure(delete_space, 0, 1)
        )
        
        # The FST accepts both email and protocol formats
        self.fst = self.delete_tokens(email | protocol).optimize()
        # self.fst = pynutil.add_weight(email_graph, 0.0) | pynutil.add_weight(url_graph, 1.0)
        # self.fst = self.delete_tokens(protocol).optimize()
        # self.fst = self.delete_tokens(email).optimize()
        # self.fst = protocol.optimize()

# tagged = 'electronic { url: "1-2-3.tv" }'
# tagged = 'electronic { url: "impressum@congstar.de" }'
# print(rewrite.top_rewrite(tagged, ElectronicFst().fst))
     

# tagged = 'electronic { url: "www.example.com" }'
# tagged = 'electronic { url: "impressum@congstar.de" }'
# tagged = 'electronic { username: "alice" domain: "uni-heidelberg.de" }'
# # tagged = tokens { electronic { username: "alice" domain: "uni-heidelberg.de" } }
# # electronic { username: "alice" domain: "uni-heidelberg.de" }
# tagged =  'electronic { username: "bob" domain: "gmail.com" }'
# print("tagged:", tagged)
# print("ElectronicFst input symbols:", ElectronicFst().fst.input_symbols())
# verbalized = rewrite.top_rewrite(tagged, ElectronicFst().fst)  # Verbalizer output
# # print(verbalized)
# # print(rewrite.rewrites(tagged, ElectronicFst().fst))

# tagged = 'electronic { url: "1-2-3.tv" }'
# print(rewrite.top_rewrite(tagged, ElectronicFst().fst))
