from mmdnn.conversion.rewriter.rewriter import UnitRewriterBase
from mmdnn.conversion.tensorflow.rewriter.gru_rewriter import GRURewriter
from mmdnn.conversion.tensorflow.rewriter.lstm_rewriter import LSTMRewriter

def process_graph(graph, weights):
    rewriter_list = [GRURewriter, LSTMRewriter]

    for rewriter in rewriter_list:
        rewriter(graph, weights).run()