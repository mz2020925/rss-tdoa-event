from ctypes import *
from os import chdir
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from RSA_API import *

from matplotlib import __version__ as __mversion__

print('Matplotlib Version:', __mversion__)
print('Numpy Version:', np.__version__)

# C:\Tektronix\RSA_API\lib\x64 needs to be added to the
# PATH system environment variable
chdir("C:\\Tektronix\\RSA_API\\lib\\x64")
rsa = cdll.LoadLibrary("RSA_API.dll")

"""################CLASSES AND FUNCTIONS################"""


def err_check(rs):
    if ReturnStatus(rs) != ReturnStatus.noError:
        raise RSAError(ReturnStatus(rs).name)


def search_connect():
    numFound = c_int(0)
    intArray = c_int * DEVSRCH_MAX_NUM_DEVICES
    deviceIDs = intArray()
    deviceSerial = create_string_buffer(DEVSRCH_SERIAL_MAX_STRLEN)
    deviceType = create_string_buffer(DEVSRCH_TYPE_MAX_STRLEN)
    apiVersion = create_string_buffer(DEVINFO_MAX_STRLEN)

    rsa.DEVICE_GetAPIVersion(apiVersion)
    print('API Version {}'.format(apiVersion.value.decode()))

    err_check(rsa.DEVICE_Search(byref(numFound), deviceIDs,
                                deviceSerial, deviceType))

    if numFound.value < 1:
        # rsa.DEVICE_Reset(c_int(0))
        print('No instruments found. Exiting script.')
        exit()
    elif numFound.value == 1:
        print('One device found.')
        print('Device type: {}'.format(deviceType.value.decode()))
        print('Device serial number: {}'.format(deviceSerial.value.decode()))
        err_check(rsa.DEVICE_Connect(deviceIDs[0]))
    else:
        # corner case
        print('2 or more instruments found. Enumerating instruments, please wait.')
        for inst in deviceIDs:
            rsa.DEVICE_Connect(inst)
            rsa.DEVICE_GetSerialNumber(deviceSerial)
            rsa.DEVICE_GetNomenclature(deviceType)
            print('Device {}'.format(inst))
            print('Device Type: {}'.format(deviceType.value))
            print('Device serial number: {}'.format(deviceSerial.value))
            rsa.DEVICE_Disconnect()
        # note: the API can only currently access one at a time
        selection = 1024
        while (selection > numFound.value - 1) or (selection < 0):
            selection = int(input('Select device between 0 and {}\n> '.format(numFound.value - 1)))
        err_check(rsa.DEVICE_Connect(deviceIDs[selection]))
    rsa.CONFIG_Preset()


"""################BLOCK IQ EXAMPLE################"""
def config_block_iq(cf=145.8e6, refLevel=0, iqBw=40e6, recordLength=10e3):
    recordLength = int(recordLength)
    rsa.CONFIG_SetCenterFreq(c_double(cf))
    rsa.CONFIG_SetReferenceLevel(c_double(refLevel))

    # 设置IQBandwidth需要研究好
    rsa.IQBLK_SetIQBandwidth(c_double(iqBw))
    # recordLength = 最大边长/3.0e8 * iqSampleRate
    rsa.IQBLK_SetIQRecordLength(c_int(recordLength))

    iqSampleRate = c_double(0)
    rsa.IQBLK_GetIQSampleRate(byref(iqSampleRate))
    # Create array of time data for plotting IQ vs time
    time = np.linspace(0, recordLength / iqSampleRate.value, recordLength)
    # time1 = []
    # step = recordLength / iqSampleRate.value / (recordLength - 1)
    # for i in range(recordLength):
    #     time1.append(i * step)
    return time


def acquire_block_iq(recordLength=10e3):
    recordLength = int(recordLength)
    ready = c_bool(False)
    iqArray = c_float * recordLength
    iData = iqArray()
    qData = iqArray()
    outLength = 0
    rsa.DEVICE_Run()
    rsa.IQBLK_AcquireIQData()
    while not ready.value:
        rsa.IQBLK_WaitForIQDataReady(c_int(100), byref(ready))
    rsa.IQBLK_GetIQDataDeinterleaved(byref(iData), byref(qData),
                                     byref(c_int(outLength)), c_int(recordLength))
    rsa.DEVICE_Stop()

    return np.array(iData) + 1j * np.array(qData)


# 将iq数据写入csv文件,但没有时间戳
def block_iq_get(cf=145.8e6, refLevel=0, iqBw=40e6, recordLength=1e3):
    # 参考函数block_iq_example(),将其中的画图代码去掉，保留首尾部分代码
    # 调用函数config_block_iq(),返回值这里用不到
    config_block_iq(cf, refLevel, iqBw, recordLength)

    # 将函数acquier_block_iq()中的语句IQ = acquier_block_iq(recordLength)换成以下代码
    recordLength = int(recordLength)
    ready = c_bool(False)
    iqArray = c_float * recordLength
    iData = iqArray()
    qData = iqArray()
    # iqData = iqArray()

    outLength = c_int(0)
    rsa.DEVICE_Run()
    rsa.IQBLK_AcquireIQData()  # 启动请求IQ块记录
    while not ready.value:
        rsa.IQBLK_WaitForIQDataReady(c_int(100), byref(ready))
    rsa.IQBLK_GetIQDataDeinterleaved(byref(iData), byref(qData), byref(outLength), c_int(recordLength))  # 把I,Q数据分开返回

    # 打印outLength和recordLengt,为什么outLength是0
    print(f'outLength:{outLength}, recordLengt:{recordLength}')

    # 打印第一个样本的时间戳
    info = IQBLK_ACQINFO()
    rsa.IQBLK_GetIQAcqInfo(byref(info))
    sample0Timestamp = info.sample0Timestamp  # 第一个样本的时间戳
    triggerSampleIndex = info.triggerSampleIndex  # trigger发生的在哪个点
    triggerTimestamp = info.triggerTimestamp
    acqStatus = info.acqStatus  # 过程中是否有错误事件发生
    print(f'第一个样本的时间戳:{sample0Timestamp}')

    # 将第一个样本的时间戳转换为时间(秒组件+纳秒组件)并打印
    o_timeSec = c_longlong(0)
    o_timeNsec = c_longlong(0)
    rsa.REFTIME_GetTimeFromTimestamp(c_longlong(i_timeStamp), byref(o_timeSec), byref(o_timeNsec))
    # print(f'秒组件:{o_timeSec.value}，纳秒组件:{o_timeNsec.value}')

    # 将秒组件+纳秒组件处理之后转换为标准时间
    stdTimeSec = o_timeSec.value + 28800
    dataArray = datetime.datetime.utcfromtimestamp(int(stdTimeSec))
    stdTimeNsec = o_timeNsec.value / 1e9
    stdTime = dataArray.strftime("%Y-%m-%d %H:%M:%S")

    rsa.DEVICE_Stop()

    # 打印刚刚获取的np.array(iData),np.array(qData)或np.array(iqData)
    # iList = np.array(iData)
    # qList = np.array(qData)
    # print(len(iList))
    # print(len(qList))
    # print(f'iqList长度:{len(iqData)},iqList末尾10组数据:{iqData[-10:]}')
    # iqList = np.array(iqData)
    # print(f'iqList长度:{len(iqList)},iqList末尾10组数据:{iqList[-10:]}')

    iqList = [(iData[x], qData[y]) for x in range(recordLength) for y in range(recordLength) if x == y]
    iqList_np = np.array(iqList)
    # print(len(iqList_np))
    # print(iqList_np[-5:])

    # 将iqData数据写入csv，此时文件中不包含时间戳
    # --想法--:知道第一个样本的时间戳，如果可以根据采集速率推算出每个样本的时间戳，再加入csv文件，就好了
    print('开始写入数据到csv')
    with open('C:\\RSA306\\RSA_API-master\\Python_zmz\\outdata\\iq_with_timestamp0808.csv', 'w', encoding='utf-8',
              newline='') as f:
        writer = csv.writer(f)
        writer.writerow([stdTime + str(stdTimeNsec)[1:]])
        writer.writerow(['Idata', 'Qdata'])
        for i in iqList_np:
            writer.writerow(i)
    print('完成IQ数据写入')

    rsa.DEVICE_Disconnect()  # 断开设备


"""################IQ STREAMING EXAMPLE################"""


def config_iq_stream(cf=1e9, refLevel=0, bw=10e6, fileDir='C:\\SignalVu-PC Files',
                     fileName='iq_stream_test', dest=IQSOUTDEST.IQSOD_FILE_SIQ,
                     suffixCtl=IQSSDFN_SUFFIX_NONE,
                     dType=IQSOUTDTYPE.IQSODT_INT16,
                     durationMsec=100):
    filenameBase = fileDir + '\\' + fileName
    bwActual = c_double(0)
    sampleRate = c_double(0)
    rsa.CONFIG_SetCenterFreq(c_double(cf))
    rsa.CONFIG_SetReferenceLevel(c_double(refLevel))

    rsa.IQSTREAM_SetAcqBandwidth(c_double(bw))
    rsa.IQSTREAM_SetOutputConfiguration(dest, dType)
    rsa.IQSTREAM_SetDiskFilenameBase(c_char_p(filenameBase.encode()))
    rsa.IQSTREAM_SetDiskFilenameSuffix(suffixCtl)
    rsa.IQSTREAM_SetDiskFileLength(c_int(durationMsec))
    rsa.IQSTREAM_GetAcqParameters(byref(bwActual), byref(sampleRate))
    rsa.IQSTREAM_ClearAcqStatus()


def iqstream_status_parser(iqStreamInfo):
    # This function parses the IQ streaming status variable
    status = iqStreamInfo.acqStatus
    if status == 0:
        print('\nNo error.\n')
    if bool(status & 0x10000):  # mask bit 16
        print('\nInput overrange.\n')
    if bool(status & 0x40000):  # mask bit 18
        print('\nInput buffer > 75{} full.\n'.format('%'))
    if bool(status & 0x80000):  # mask bit 19
        print('\nInput buffer overflow. IQStream processing too slow, ',
              'data loss has occurred.\n')
    if bool(status & 0x100000):  # mask bit 20
        print('\nOutput buffer > 75{} full.\n'.format('%'))
    if bool(status & 0x200000):  # mask bit 21
        print('Output buffer overflow. File writing too slow, ',
              'data loss has occurred.\n')


def iq_stream_example():
    print('\n\n########IQ Stream Example########')
    search_connect()

    bw = 40e6
    dest = IQSOUTDEST.IQSOD_FILE_SIQ_SPLIT
    durationMsec = 100
    waitTime = 0.1
    iqStreamInfo = IQSTREAM_File_Info()

    complete = c_bool(False)
    writing = c_bool(False)

    config_iq_stream(bw=bw, dest=dest, durationMsec=durationMsec)

    rsa.DEVICE_Run()
    rsa.IQSTREAM_Start()
    while not complete.value:
        sleep(waitTime)
        rsa.IQSTREAM_GetDiskFileWriteStatus(byref(complete), byref(writing))
    rsa.IQSTREAM_Stop()
    print('Streaming finished.')
    rsa.IQSTREAM_GetFileInfo(byref(iqStreamInfo))
    iqstream_status_parser(iqStreamInfo)
    rsa.DEVICE_Stop()
    rsa.DEVICE_Disconnect()
