from handleFrame import readUniqueData, convertToL2, saveData

uniqueData = readUniqueData()
l2Data = convertToL2(uniqueData)
saveData(l2Data, "dataUniqueL2.csv")
