import React, {useState, useRef, useEffect} from 'react';
import './App.css';
// RCE CSS
import 'react-chat-elements/dist/main.css';
import {MessageList, Input, Button} from 'react-chat-elements'

const axios = require('axios');

const globalMessageData = [{
    position: 'left',
    type: 'text',
    text: 'Hello, my dear friend, this is GZZchat, a transformer backed chatbot',
    date: new Date()
}];

function App() {
    let [messageData, setMessageData] = useState(globalMessageData);
    let [input, setInput] = useState('');
    let inputRef = useRef();

    useEffect(() => {
        let interval = setInterval(() => {
            setMessageData(globalMessageData.slice(-10000))
        }, 10000);
        return () => {
            clearInterval(interval)
        }
    });

    let onSend = () => {
        let text = input.trim();
        if (text === '') {
            return
        }
        globalMessageData.push({
            position: 'right',
            type: 'text',
            text: text,
            date: new Date()
        });
        setMessageData([...globalMessageData]);
        axios.get(`/reply/${input}`).then((response) => {
            globalMessageData.push({
                position: 'left',
                type: 'text',
                text: response.data,
                date: new Date()
            });
            setMessageData([...globalMessageData]);
        });
        inputRef.current.clear()
    };

    return (
        <div className="App">
            <div className="main">
                <MessageList
                    className='message-list'
                    lockable={true}
                    toBottomHeight={'100%'}
                    dataSource={messageData}/>
                <Input
                    className='input-box'
                    placeholder="Type here..."
                    multiline={true}
                    onChange={(event) => {
                        setInput(event.target.value)
                    }}
                    onKeyPress={(e) => {
                        if (e.shiftKey && e.charCode === 13) {
                            return true;
                        }
                        if (e.charCode === 13) {
                            onSend();
                            e.preventDefault();
                            return false;
                        }
                    }}
                    ref={inputRef}
                    rightButtons={
                        <Button
                            color='white'
                            backgroundColor='black'
                            text='Send'
                            onClick={onSend}/>
                    }/>
            </div>
        </div>
    );
}


export default App;
