from pkg_resources import parse_version
import kaitaistruct
import pickle
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO
from random import randrange


if parse_version(kaitaistruct.__version__) < parse_version('0.9'):
    raise Exception("Incompatible Kaitai Struct Python API: 0.9 or later is required, but you have %s" % (kaitaistruct.__version__))

class Datasheet(KaitaiStruct):
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self._raw_header = self._io.read_bytes(60)
        _io__raw_header = KaitaiStream(BytesIO(self._raw_header))
        self.header = Datasheet.DatasheetHeader(_io__raw_header, self, self._root)
        self._raw_meta = self._io.read_bytes(32)
        _io__raw_meta = KaitaiStream(BytesIO(self._raw_meta))
        self.meta = Datasheet.DatasheetMeta(_io__raw_meta, self, self._root)
        self.columns = [None] * (self.meta.column_count)
        for i in range(self.meta.column_count):
            self.columns[i] = Datasheet.DatasheetColumns(self._io, self, self._root)

        self.rows = [None] * (self.meta.row_count)
        for i in range(self.meta.row_count):
            self.rows[i] = Datasheet.DatasheetRows(self._io, self, self._root)


    class DatasheetHeader(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.version = self._io.read_u4le()
            self.unknown1 = self._io.read_u4le()
            self.unique_id_offset = self._io.read_u4le()
            self.unknown2 = self._io.read_u4le()
            self.data_type_offset = self._io.read_u4le()
            self.row_number = self._io.read_u4le()
            self.plain_text_length = self._io.read_u4le()
            self.unknown3 = self._io.read_u4le()
            self.unknown4 = self._io.read_u4le()
            self.unknown5 = self._io.read_u4le()
            self.unknown6 = self._io.read_u4le()
            self.unknown7 = self._io.read_u4le()
            self.unknown8 = self._io.read_u4le()
            self.unknown9 = self._io.read_u4le()
            self.plain_text_offset = self._io.read_u4le()

        @property
        def unique_id(self):
            if hasattr(self, '_m_unique_id'):
                return self._m_unique_id if hasattr(self, '_m_unique_id') else None

            io = self._root._io
            _pos = io.pos()
            io.seek(((self.unique_id_offset + self.plain_text_offset) + 60))
            self._m_unique_id = (io.read_bytes_term(0, False, True, True)).decode(u"UTF-8")
            io.seek(_pos)
            return self._m_unique_id if hasattr(self, '_m_unique_id') else None

        @property
        def type(self):
            if hasattr(self, '_m_type'):
                return self._m_type if hasattr(self, '_m_type') else None

            io = self._root._io
            _pos = io.pos()
            io.seek(((self.data_type_offset + self.plain_text_offset) + 60))
            self._m_type = (io.read_bytes_term(0, False, True, True)).decode(u"UTF-8")
            io.seek(_pos)
            return self._m_type if hasattr(self, '_m_type') else None


    class DatasheetMeta(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.crc32 = self._io.read_u4be()
            self.unknown10 = self._io.read_u4le()
            self.column_count = self._io.read_u4le()
            self.row_count = self._io.read_u4le()
            self.unknown11 = self._io.read_u4le()
            self.unknown12 = self._io.read_u4le()
            self.unknown13 = self._io.read_u4le()
            self.unknown14 = self._io.read_u4le()


    class DatasheetRow(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.value_offset = self._io.read_u4le()
            self.value_duplicate = self._io.read_u4le()

        @property
        def value(self):
            if hasattr(self, '_m_value'):
                return self._m_value if hasattr(self, '_m_value') else None

            io = self._root._io
            _pos = io.pos()
            io.seek(((self.value_offset + self._root.header.plain_text_offset) + self._root.header._io.size()))
            self._m_value = (io.read_bytes_term(0, False, True, True)).decode(u"UTF-8")
            io.seek(_pos)
            return self._m_value if hasattr(self, '_m_value') else None


    class DatasheetRows(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.row = [None] * (self._root.meta.column_count)
            for i in range(self._root.meta.column_count):
                self.row[i] = Datasheet.DatasheetRow(self._io, self, self._root)



    class DatasheetColumns(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.unknown15 = self._io.read_u4le()
            self.column_name_offset = self._io.read_u4le()
            self.column_type = self._io.read_u4le()

        @property
        def value(self):
            if hasattr(self, '_m_value'):
                return self._m_value if hasattr(self, '_m_value') else None

            io = self._root._io
            _pos = io.pos()
            io.seek(((self.column_name_offset + self._root.header.plain_text_offset) + self._root.header._io.size()))
            self._m_value = (io.read_bytes_term(0, False, True, True)).decode(u"UTF-8")
            io.seek(_pos)
            return self._m_value if hasattr(self, '_m_value') else None

class LootBucketEntry():
    def __init__(self, item_name, bucket_name, zone_minlevel_req, zone_maxlevel_req, tags, item_minqty, item_maxqty):
        self.item_name = item_name
        self.bucket_name = bucket_name
        self.zone_minlevel_req = zone_minlevel_req
        self.zone_maxlevel_req = zone_maxlevel_req
        self.tags = tags
        self.item_minqty = item_minqty
        self.item_maxqty = item_maxqty

    def PrintAttributes(self):
        print("lootbucket entry", "\nItem Name: ", self.item_name, "\n LootBucket Name: ", self.bucket_name, "\nZone Level Requirement", self.zone_minlevel_req, "-", self.zone_maxlevel_req, "\ntags: ", self.tags, "\nQuantity: ", self.item_minqty, "-", self.item_maxqty)

class LootTableEntry():
    def __init__(self, loot_table_id, and_or, lucksafe, maxroll, items, conditions):
        self.loot_table_id = loot_table_id
        self.and_or = and_or
        self.lucksafe = lucksafe
        self.maxroll = maxroll
        self.items = items
        self.conditions = conditions

    def PrintAttributes(self):
        print("LootTable Object", "\nTable ID: ", self.loot_table_id, "\n AND/OR: ", self.and_or, "\nLuckSafe: ", self.lucksafe, "\nMaxRoll: ", self.maxroll, "\nConditions:")
        #for condition in self.conditions:
        #    print(condition)
        print(self.conditions)
        print("\n Items:")
        for item in self.items:
            item.PrintAttributes()

class LootTableItem():
    def __init__(self, item_name, item_minqty, item_maxqty, item_prob, parent_loottable):
        self.item_name = item_name
        self.item_minqty = item_minqty
        self.item_maxqty = item_maxqty
        self.item_prob = item_prob
        self.parent_loottable = parent_loottable

    def PrintAttributes(self):
        print("LootTable Item", "\nItem Name: ", self.item_name, "\nItem Quantity: ", self.item_minqty, "-", self.item_maxqty, "\nItemProbability: ", self.item_prob, "(", (self.parent_loottable.maxroll - self.item_prob) / self.parent_loottable.maxroll * 100, "% base chance)")

class ProbabilityObj():
    def __init__(self, name, probability, quantmin, quantmax):
        self.name = name
        self.probability = probability
        self.quantity = [quantmin, quantmax]

def ExtractLootTableEntries(filesource):

    filesource = Datasheet.from_file(filesource)

    columncounter = 0
    loot_table_objects = []
    current_loottable = LootTableEntry("", "AND", False, 100000, [], [])
    mode = "lootmain"
    loottableid = ""
    itemcounter = 0

    for group in filesource.rows:
        for entry in group.row:

            columnvalue = filesource.columns[columncounter].value
           
            if columnvalue == "LootTableID":
                if entry.value.find("_Qty") != -1:
                    mode = "lootqty"
                    itemcounter = 0
                elif entry.value.find("_Probs") != -1:
                    mode = "lootprobs"
                    itemcounter = 0
                else:
                    mode = "lootmain"
                    if current_loottable.loot_table_id != "":
                        loot_table_objects.append(current_loottable)
                    current_loottable = LootTableEntry(entry.value, "AND", False, 100000, [], [])
            if columnvalue == "AND/OR":
                if mode == "lootmain":
                    current_loottable.and_or = entry.value
            if columnvalue == "LuckSafe":
                if mode == "lootmain":
                    if entry.value != "TRUE":
                        current_loottable.lucksafe = False
                    else:
                        current_loottable.lucksafe = True
            if columnvalue == "MaxRoll":
                if mode == "lootprobs":
                    if entry.value == "":
                        current_loottable.maxroll = 100000
                    else:
                        current_loottable.maxroll = int(entry.value)
            if columnvalue == "Conditions":
                if mode == "lootmain":
                    if "," in entry.value:
                        commaindex = entry.value.find(",")
                        current_loottable.conditions.append(entry.value[:commaindex])
                        current_loottable.conditions.append(entry.value[commaindex+1:])
                    else:
                        current_loottable.conditions.append(entry.value)
            if "Item" in columnvalue:

                if mode == "lootmain" and entry.value != "":
                    current_loottable.items.append(LootTableItem(entry.value, None, None, None, current_loottable))

                if mode == "lootqty" and entry.value != "":
                    currentitem = current_loottable.items[itemcounter]
                    if "-" not in entry.value:
                        currentitem.item_minqty = int(entry.value)
                        currentitem.item_maxqty = int(entry.value)
                    else:
                        currentitem.item_minqty = int(entry.value[:entry.value.find("-")])
                        currentitem.item_maxqty = int(entry.value[entry.value.find("-") + 1:])

                if mode == "lootprobs" and entry.value != "":
                    current_loottable.items[itemcounter].item_prob = int(entry.value)

                itemcounter = itemcounter + 1

            columncounter = columncounter + 1
        columncounter = 0

    return loot_table_objects




def ExtractLootBucketEntries(filesource):

    filesource = Datasheet.from_file(filesource)
    bucketnamelist = []
    lootbucketentries = []

    counter = 0
    for entry in filesource.columns:
        if(entry.value.startswith("LootBucket")):
            bucketnamelist.append(filesource.rows[0].row[counter].value)

        counter = counter + 1

    counter = 4
    lootbucketcounter = 0
    bucketname = ""
    itemname = ""
    tags = []
    minlevelrequirement = 0
    maxlevelrequirement = 70
    minquantity = 1
    maxquantity = 1

    for group in filesource.rows:
        for entry in group.row:
            if entry.value == "FIRSTROW" or "DATA" in entry.value:
                lootbucketcounter = 0
                counter = 4
            elif counter == 4:
                
                bucketname = bucketnamelist[lootbucketcounter]
                lootbucketcounter = lootbucketcounter + 1
                counter = 0

            #parse tags
            elif counter == 0:

                if(entry.value == ""):
                    moretags = False
                    tags = []
                    minlevelrequirement = 0
                    maxlevelrequirement = 70
                else:
                    moretags = True
                    tags = []
                
                tagstring = entry.value

                while moretags == True:
                    
                    thistag = ""
                    commaindex = tagstring.find(",")

                    if commaindex != -1:
                        thistag = tagstring[:commaindex]
                        tagstring = tagstring[commaindex+1:]
                    else:
                        thistag = tagstring
                        moretags = False

                    #parse level requirement
                    if thistag.find("MinContLevel:") != -1:

                        dashindex = thistag.find("-")

                        if dashindex == -1:
                            minlevelrequirement = int(thistag[thistag.find(":")+1:])
                            maxlevelrequirement = 70
                        else:
                            minlevelrequirement = int(thistag[thistag.find(":")+1:dashindex])
                            maxlevelrequirement = int(thistag[dashindex+1:])
                    
                    elif thistag.find("Level") != -1:
                        
                        dashindex = thistag.find("-")

                        if dashindex == -1:
                            minlevelrequirement = int(thistag[thistag.find(":")+1:])
                            maxlevelrequirement = 70
                        else:
                            minlevelrequirement = int(thistag[thistag.find(":")+1:dashindex])
                            maxlevelrequirement = int(thistag[dashindex+1:])

                    else:
                        tags.append(thistag)
                        levelrequirement = 0

                counter = counter + 1


            elif counter == 2:

                itemname = entry.value
                counter = counter + 1

            elif counter == 3:

                if "-" not in entry.value:
                    minquantity = 1
                    maxquantity = 1
                else:
                    minquantity = int(entry.value[:entry.value.find("-")])
                    maxquantity = int(entry.value[entry.value.find("-")+1:])

                if itemname != "":
                    lootbucketentries.append(LootBucketEntry(itemname, bucketname, minlevelrequirement, maxlevelrequirement, tags, minquantity, maxquantity))
                counter = counter + 1

            else:
                counter = counter + 1

            
            
    
    return lootbucketentries

def GetLootTableById(tableid):

    if "[LTID]" in tableid:
        tableid = tableid[6:]

    for table in loot_table_db:
        if table.loot_table_id == tableid:
            return table
    return None

def GetLootBucketById(lootbucketid, tags, level):

    if "[LBID]" in lootbucketid:
        lootbucketid = lootbucketid[6:]

        elligible_items = []
        for item in loot_bucket_db:

            elligible = True
            if (item.bucket_name == lootbucketid) and (level >= item.zone_minlevel_req) and (level <= item.zone_maxlevel_req):

                if len(item.tags) > 0:
                    elligible = False
                    for tag in tags:
                        if tag in item.tags:
                            elligible = True

                
                if elligible == True:
                    elligible_items.append(item)
    
    return elligible_items

def PullItemFromBucket(lootbucketid, tags, level):

    if "[LBID]" in lootbucketid:
        lootbucketid = lootbucketid[6:]

    elligible_items = []
    for item in loot_bucket_db:
        elligible = True
        if (item.bucket_name == lootbucketid) and (level >= item.zone_minlevel_req) and (level <= item.zone_maxlevel_req):
            
            if len(item.tags) > 0:
                    elligible = False
                    for tag in tags:
                        if tag in item.tags:
                            elligible = True
                
            if elligible == True:
                elligible_items.append(item)
    
    return elligible_items[randrange(len(elligible_items))]


                    
def CreateProbabilityTreeFromTable(table, luck = 0, level = 70, tags = [], tabletype = "loottable", baseprobability = 1.0, fishrollmod = 0):

    probslist = []
    if tabletype == "lootbucket":
        for entry in table:
            baseprob = 1/len(table)
            if baseprob > 1:
                 baseprob = 1
            probslist.append(ProbabilityObj(entry.item_name, baseprob*baseprobability, entry.item_minqty, entry.item_maxqty))
       
    if tabletype == "loottable":

        for entry in table.items:

            
            elligible = True
            #parse table conditions, ignoring the ones that call for specific areas/tags and removing the ones that have to do with roll logic
            #if tags don't match the condition, the item is not elligible
            specialconditions = table.conditions
            conditions_to_ignore = ["GlobalMod", "MinPOIContLevel", "EnemyLevel", "Level", "FishRarity", "FishSize", ""]
            for condition in specialconditions:
                if condition not in conditions_to_ignore and condition not in tags:
                    elligible = False

            #base case if the item is not elligible or otherwise cannot be calculated
            probability = 0

            if elligible == True:

                rollmodifier = 0
                maxroll = table.maxroll
                rollreq = entry.item_prob

                if "Level" in table.conditions or "MinPOIContLevel" in table.conditions or "EnemyLevel" in table.conditions:
                    rollmodifier = level
                elif "FishSize" in table.conditions:
                    rollmodifier = 0
                elif "FishRarity" in table.conditions:
                    rollmodifier = fishrollmod * 10000
                    maxroll = maxroll - fishrollmod * 10000
                else:
                    rollmodifier = luck


                
                if table.lucksafe != True or "FishSize" in table.conditions or "FishRarity" in table.conditions:
                     rollreq = rollreq - rollmodifier


                        


                if table.and_or == "AND":
                    #handling maxroll = 0 (so we don't divide by 0)
                    if maxroll == 0 and rollreq > 0:
                        probability = 0
                    elif maxroll == 0 and rollreq <= 0:
                        probability = 1
                    #take the simple probability of rolling higher than the item's threshold, clamping between 0 and 1
                    else:
                        baseprob = (maxroll - rollreq) / maxroll
                        if baseprob > 1:
                            baseprob = 1
                        elif baseprob < 0:
                            baseprob = 0
                        probability = baseprob

                elif table.and_or == "OR":
                    
                    #tracker values
                    number_of_equal_probs = 1
                    list_of_greater_probs = []

                    #scan table for entries with equal roll values and higher roll values
                    for item in table.items:
                        if item != entry:
                            comparison_rollreq = item.item_prob

                            if table.lucksafe != True or "FishRarity" in table.conditions:
                                comparison_rollreq = comparison_rollreq - rollmodifier

                            if comparison_rollreq == rollreq:
                                number_of_equal_probs = number_of_equal_probs + 1
                            elif comparison_rollreq > rollreq:
                                list_of_greater_probs.append(comparison_rollreq)

                    #compare selected entry with the next highest roll value (use the maximum roll eg 100,000 usually if there is 
                    #no higher value
                    if len(list_of_greater_probs) > 0:
                        list_of_greater_probs.sort()
               
                    if rollreq < 0:
                        rollreq = 0
                    
                    #handling maxroll = 0 (so we don't divide by 0)
                    if maxroll == 0 and rollreq > 0:
                        probability = 0
                    elif maxroll == 0 and rollreq <= 0:
                        if len(list_of_greater_probs) > 0 and list_of_greater_probs[0] <= 0:
                            probability = 0
                        else:
                            probability = 1 / number_of_equal_probs
                    # if there are no higher items to account for, or the higher items are not eligible to drop, take prob as normal
                    elif len(list_of_greater_probs) < 1 or list_of_greater_probs[0] > maxroll:
                        baseprob = (maxroll - rollreq) / maxroll
                        if baseprob > 1:
                            baseprob = 1
                        elif baseprob < 0:
                            baseprob = 0
                        probability = baseprob / number_of_equal_probs
                    #When two roll requirements are negative, the higher one will win 100% of the time so this will drop 0% of the time
                    elif list_of_greater_probs[0] <= 0:
                        probability = 0
                    #In all other cases, subtract the probability of rolling above this item bracket from the total probability
                    #Split the odds with all other items that have the same threshold
                    else:
                        baseprob = (list_of_greater_probs[0] - rollreq) / maxroll
                        if baseprob > 1:
                            baseprob = 1
                        elif baseprob < 0:
                            baseprob = 0
                        probability = baseprob / number_of_equal_probs
                
            probability = probability * baseprobability
            print(probability)
            probslist.append(ProbabilityObj(entry.item_name, probability, entry.item_minqty, entry.item_maxqty))

            if "[LTID]" in entry.item_name:
                print("found new table", entry.item_name)
                probslist.extend(CreateProbabilityTreeFromTable(GetLootTableById(entry.item_name), luck, level, tags, "loottable", probability, fishrollmod))
            elif "[LBID]" in entry.item_name:
                print("found new table", entry.item_name)
                tags.append(table.loot_table_id)
                probslist.extend(CreateProbabilityTreeFromTable(GetLootBucketById(entry.item_name, tags, level), luck, level, tags, "lootbucket", probability, fishrollmod))
            
    return probslist


def RollOnTable(table, luck, level, tags):

    global itemcounttracker

    fullyrolled = False
    final_loot = []
    tables_queued = []

    while fullyrolled == False:

        print("rolling on table _", table.loot_table_id, "_")
        
        roll = randrange(table.maxroll) + 1
        if table.lucksafe == False:
            roll = roll + luck
        print("rolled:  ", roll)

        itemsrolled = []

        for item in table.items:
            if item.item_prob <= roll:
                itemsrolled.append(item)
        if len(itemsrolled) > 0:
            if table.and_or == "OR":
                itemsrolled.sort(key = lambda x:x.item_prob)
                highestvalue = itemsrolled[-1].item_prob

                itemraffle = []
                for item in itemsrolled:
                    if item.item_prob >= highestvalue:
                        itemraffle.append(item)
                rafflewinner = itemraffle[randrange(len(itemraffle))]
            
                if "[LTID]" in rafflewinner.item_name:
                    tables_queued.append(rafflewinner.item_name[6:])
                elif "[LBID]" in rafflewinner.item_name:
                    ####
                    if rafflewinner.item_name == "[LBID]ContainerTrophyArtifacts":
                        print("OMG TROPHY\nOMG TROPHY\nOMG TROPHY\nOMG TROPHY\nOMG TROPHY\n")
                        itemcounttracker = itemcounttracker + 1
                    ####
                    final_loot.append(PullItemFromBucket(rafflewinner.item_name, tags, level))
                else:
                    final_loot.append(rafflewinner)

            else:
                for item in itemsrolled:
                    if "[LTID]" in item.item_name:
                        tables_queued.append(item.item_name[6:])
                    elif "[LBID]" in item.item_name:
                        ####
                        if item.item_name == "[LBID]ContainerTrophyArtifacts":
                            print("OMG TROPHY\nOMG TROPHY\nOMG TROPHY\nOMG TROPHY\nOMG TROPHY\n")
                            itemcounttracker = itemcounttracker + 1
                        ####
                        final_loot.append(PullItemFromBucket(item.item_name, tags, level))
                    else:
                        final_loot.append(item)

        if len(tables_queued) > 0:
            tableid = tables_queued.pop(-1)
            table = GetLootTableById(tableid)
        else:
            fullyrolled = True

    print("Items received:")
    for item in final_loot:
        print(item.item_minqty, "-", item.item_maxqty, " ", item.item_name)


def DecipherDatasheet(filesource, fileoutput):

    filesource = Datasheet.from_file(filesource)
    fileoutput = open(fileoutput, "w")

    datacolumns = []

    for entry in filesource.columns:

        datacolumns.append(entry.value)

    counter = 0

    for entry in filesource.rows:
        for thing in entry.row:

            fileoutput.write(datacolumns[counter])
            fileoutput.write("  ")
            fileoutput.write(thing.value)
            fileoutput.write("\n")

            counter = counter + 1
            if counter >= len(datacolumns):
                counter = 0
                fileoutput.write("___________________________________________\n")

#DecipherDatasheet("javelindata_lootbuckets.datasheet", "loot_buckets_readable.txt")

#loot_table_db = ExtractLootTableEntries("javelindata_loottables.datasheet")
#pickle.dump(loot_table_db, open("pythontables.pickle", "wb"))
#loot_bucket_db = ExtractLootBucketEntries("javelindata_lootbuckets.datasheet")
#pickle.dump(loot_bucket_db, open("pythonbuckets.pickle", "wb"))



loot_table_db = pickle.load(open("pythontables.pickle", "rb"))

loot_bucket_db = pickle.load(open("pythonbuckets.pickle", "rb"))

#obj = GetLootTableById#obj.PrintAttributes()



itemcounttracker = 0

luck = 0
FishingLuck = 0
id = "CreatureLootMaster"
tablelevel = 60
baseprobability = 1.0
tags = ["Dungeon", "Elite", "Reekwater00"]

probs = CreateProbabilityTreeFromTable(GetLootTableById(id), luck, tablelevel, tags, "loottable", baseprobability, FishingLuck)

filename = id + str(luck) + "luck_Level" + str(tablelevel) + ".txt"
f = open(filename, "w")

#counter = 0.0
f.write("LUCK: " + str(luck) + "\n" + "FISHINGLUCK: " + str(FishingLuck) + "\n")
GlobalNamedList_TurnOffSwitch = False
for items in probs:

    if  ("[LTID]" in items.name or "[LBID]" in items.name or items.probability != 0) and GlobalNamedList_TurnOffSwitch == False:

        quantstring = ""

        if items.quantity[0] == items.quantity[1]:
            quantstring = str(items.quantity[1])
        else:
            quantstring = str(items.quantity[0]) + "-" + str(items.quantity[1])

        if "[LTID]" in items.name or "[LBID]" in items.name:
            leadingstring = quantstring + " "
        else:
            leadingstring = "      " + quantstring + " "

        if items.probability < 0.01 and items.probability > 0:
            f.write(leadingstring + items.name + ":" + "{:.5f}".format(items.probability*100) + "%\n")
        elif items.probability > 0:
            f.write(leadingstring + items.name + ":" + str(round(items.probability*100, 3)) + "%\n")

    if GlobalNamedList_TurnOffSwitch == False and (items.name == "[LBID]GlobalNamedList" or items.name == "[LBID]RecoveryItems"):
        GlobalNamedList_TurnOffSwitch = True
    elif GlobalNamedList_TurnOffSwitch == True:
        if "[LTID]" in items.name or "[LBID]" in items.name:
            GlobalNamedList_TurnOffSwitch = False

 
#print(counter)
#for i in range(0, 10000000):
#    print("\n__________")
    #RollOnTable(GetLootTableById("TreeLarge") , 3500, 60, [])
#    print("__________\n")

#print("with 3,500 luck you got", itemcounttracker, "trophies in 10,000,000 drops (that's a ", itemcounttracker / 10000000, "drop rate,")


