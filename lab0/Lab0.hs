{-# LANGUAGE OverloadedStrings #-}

import Control.Monad
import Network.HTTP.Conduit
import Network.HTTP.Client (defaultManagerSettings)
import Data.Text.Encoding
import qualified Data.ByteString.Lazy as L
import qualified Data.ByteString.Char8 as C
import Network (withSocketsDo)

(email, name) = ("", encodeUtf8 "") -- Адрес почты и фамилия с инициалами
university = ["MEPhI", "MIPT"]!!2 -- ВУЗ ( MEPhI | MIPT) 
group = encodeUtf8 "" -- Группа

pascal :: Int -> Int -> Int
pascal c r = 1 -- а тут решение

printIt :: Int -> C.ByteString
printIt n = C.pack $ show $ [pascal y x | x <- [0..n], y <- [0..x]]

main :: IO()
main = 
  withSocketsDo $ do
  initReq <- parseRequest "POST http://91.239.142.110:13666/lab0"
  let req = urlEncodedBody [("email", email), ("name", name), ("university", university), ("group", group), ("content", printIt 20)] $ initReq
  manager <- newManager defaultManagerSettings
  response <- httpLbs req manager
  L.putStr $ responseBody response
