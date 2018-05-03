{-# LANGUAGE OverloadedStrings #-}

import System.IO
import System.Environment
import System.Directory
import Control.Monad
import qualified Data.ByteString.Lazy.Char8 as L
import Data.Text.Encoding
import Network.HTTP.Conduit
import qualified Data.Text as T
import Text.HTML.DOM (parseLBS)
import Text.XML.Cursor (Cursor, attributeIs, content, element, fromDocument, child, ($//), (&|), (&//), (&/), (>=>)) 
import Network (withSocketsDo)

-- почтовый адрес
email = ""

cursorFor :: String -> IO Cursor
cursorFor u = do
     page <- withSocketsDo $ simpleHttp u
     return $ fromDocument $ parseLBS page

lab3 :: IO [T.Text]
lab3 = do
  cursor <- cursorFor "http://mipt.ru/diht/bases/"
  return $ cursor $// element "ul" >=> attributeIs "class" "right-menu" &// element "a" >=> child &| T.concat . content

main :: IO()
main = withSocketsDo $ do
  dir <- getCurrentDirectory
  initReq <- parseRequest "POST http://91.239.142.110:13666/lab3"
  handle <- openFile (dir ++ "/Lab3.hs") ReadMode
  hSetEncoding handle utf8_bom
  content <- hGetContents handle
  let req = urlEncodedBody [("email", email), ("content", encodeUtf8 $ T.pack content) ] $ initReq
  manager <- newManager defaultManagerSettings
  response <- httpLbs req manager
  hClose handle
  L.putStrLn $ responseBody response
