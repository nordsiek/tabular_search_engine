import argparse
import warnings

from request_engine import PreviewModel

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-o_key", "--openai_api_key", help="API key from OpenAI.", dest="openai_api_key", required=True)
parser.add_argument("-b_key", "--bing_api_key", help="API key from Bing.", dest="bing_api_key", required=True)
parser.add_argument("-p_key", "--pinecone_api_key", help="API key from Pinecone.", dest="pinecone_api_key", required=False)
parser.add_argument("-db", "--database", help="Vector DB storage type.", dest='dbType', required=False)
parser.add_argument("-n", "--num_pages", help="Specify the number of web pages to retrieve.", dest="num_pages", required=False)
parser.add_argument("-m", "--model", help="Select alternative OpenAI model", dest="model", required=False)
parser.add_argument("-v", "--verbose", help="Print additional debug information", dest="verbose", required=False)
parser.add_argument("-r", "--request", help="Your request.", dest="rows", required=True, nargs='+')
parser.add_argument("-a", "--again", help="Retry with reloading results.", dest="redo", required=False)
parser.add_argument("-add", "--add_info", help="Required additional info to your request.", dest="columns",
                    required=False, nargs='+')

args = parser.parse_args()

pm = PreviewModel(**vars(args))
pm.run()
